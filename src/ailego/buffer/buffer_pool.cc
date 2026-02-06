#include <zvec/ailego/buffer/buffer_pool.h>

namespace zvec {
namespace ailego {

void Counter::record(const std::string &name, int64_t value) {
	auto it = static_counters.find(name);
	if (it == static_counters.end()) {
			auto counter = std::make_unique<std::atomic<int64_t>>(0);
			it = static_counters.emplace(name, std::move(counter)).first;
	}
	it->second->fetch_add(value);
}

void Counter::display() {
	for (const auto &pair : static_counters) {
		std::cout << pair.first << ": " << pair.second->load() << std::endl;
	}
}

int LRUCache::init(size_t block_size) {
	block_size_ = block_size;
	for(size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
		queues_.push_back(ConcurrentQueue(block_size));
	}
	return 0;
}

bool LRUCache::evict_single_block(BlockType &item) {
	// std::cerr << "dequeue: " << item.first << std::endl;
	bool found = false;
	for(size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
		found = queues_[i].try_dequeue(item);
		// std::cerr << "dequeue: " << found << std::endl;
		if(found) {
			break;
		}
	}
	return found;
}

bool LRUCache::add_single_block(const LPMap *lp_map, const BlockType &block, int block_type) {
	bool ok = queues_[block_type].try_enqueue(block);
	if(++evict_queue_insertions_ % block_size_ == 0) {
		this->clear_dead_node(lp_map);
	}
	return ok;
}

void LRUCache::clear_dead_node(const LPMap *lp_map) {
	for(int i = 0; i < CATCH_QUEUE_NUM; i++) {
		int clear_count = 0;
		ConcurrentQueue tmp(block_size_);
		BlockType item;
		while(queues_[i].try_dequeue(item) && (clear_count++ < block_size_)) {
			if(!lp_map->isDeadBlock(item)) {
				tmp.try_enqueue(item);
			}
		}
		while(tmp.try_dequeue(item)) {
			if(!lp_map->isDeadBlock(item)) {
				queues_[i].try_enqueue(item);
			}
		}
	}
}

void LPMap::init(size_t entry_num) {
	if (entries_) {
		delete[] entries_;
	}
	entry_num_ = entry_num;
	entries_ = new Entry[entry_num_];
	for (size_t i = 0; i < entry_num_; i++) {
		entries_[i].ref_count.store(std::numeric_limits<int>::min());
		entries_[i].load_count.store(0);
		entries_[i].buffer = nullptr;
	}
	cache_.init(entry_num);
}

char* LPMap::acquire_block(block_id_t block_id) {
	assert(block_id < entry_num_);
	Entry &entry = entries_[block_id];
	if (entry.ref_count.load() == 0) {
		++entry.load_count;
		// std::cout << entry.load_count.load() << std::endl;
	}
	++entry.ref_count;
	// std::cout << entry.ref_count.load() << std::endl;
	if (entry.ref_count.load() < 0) {
		// std::cout << "acquire block failed: " << block_id << ", " << entry.ref_count.load() << std::endl;
		return nullptr;
	}
	return entry.buffer;
}

void LPMap::release_block(block_id_t block_id) {
	assert(block_id < entry_num_);
	Entry &entry = entries_[block_id];
	int rc = entry.ref_count.fetch_sub(1);
	// std::cout << "release block: " << block_id << ", " << entry.ref_count.load() << std::endl;
	// assert(rc > 0);
	if(entry.ref_count.load() == 0) {
		LRUCache::BlockType block;
		block.first = block_id;
		block.second = entry.load_count.load();
		cache_.add_single_block(this, block, 0);
	}
}

char* LPMap::evict_block(block_id_t block_id) {
	// std::cout << "evict block: " << block_id << std::endl;
	assert(block_id < entry_num_);
	Entry &entry = entries_[block_id];
	int expected = 0;
	if (entry.ref_count.compare_exchange_strong(
					expected, std::numeric_limits<int>::min())) {
		char *buffer = entry.buffer;
		entry.buffer = nullptr;
		return buffer;
	} else {
		return nullptr;
	}
}

char* LPMap::set_block_acquired(block_id_t block_id, char *buffer) {
	assert(block_id < entry_num_);
	Entry &entry = entries_[block_id];
	if (entry.ref_count.load() >= 0) {
		entry.ref_count.fetch_add(1);
		// std::cout << "Set block2 " << block_id << std::endl;
		return entry.buffer;
	}
	// if (buffer == nullptr) std::cout << "Set block " << block_id << std::endl;
	entry.buffer = buffer;
	entry.ref_count.store(1);
	entry.load_count.fetch_add(1);
	return buffer;
}

void LPMap::recycle(moodycamel::ConcurrentQueue<char *> &free_buffers) {
	LRUCache::BlockType block;
	do {
		bool ok = cache_.evict_single_block(block);
		if(!ok) {
			return;
		}
	} while(isDeadBlock(block));
	// std::cout << "evict_block done: " << block.first << ", " << block.second << std::endl;
	char *buffer = evict_block(block.first);
	if (buffer) {
		free_buffers.try_enqueue(buffer);
	}
}

VecBufferPool::VecBufferPool(const std::string &filename, size_t pool_capacity, size_t block_size)
		: pool_capacity_(pool_capacity) {
	fd_ = open(filename.c_str(), O_RDONLY);
	if (fd_ < 0) {
		throw std::runtime_error("Failed to open file: " + filename);
	}
	struct stat st;
	if (fstat(fd_, &st) < 0) {
		throw std::runtime_error("Failed to stat file: " + filename);
	}
	file_size_ = st.st_size;

	size_t buffer_num = pool_capacity_ / block_size;
	size_t block_num = file_size_ / block_size + 500;
	lp_map_.init(block_num);
	for (size_t i = 0; i < buffer_num; i++) {
		char *buffer = (char *)aligned_alloc(64, block_size);
		if (buffer != nullptr) {
			bool ok = free_buffers_.try_enqueue(buffer);
			// if(!ok) std::cerr << i << std::endl;
		}
	}
	std::cout << "buffer_num: " << buffer_num << std::endl;
	std::cout << "entry_num: " << lp_map_.entry_num() << std::endl;
}

VecBufferPoolHandle VecBufferPool::get_handle() {
	return VecBufferPoolHandle(*this);
}

char* VecBufferPool::acquire_buffer(block_id_t block_id, size_t offset, size_t size, int retry) {
	char *buffer = lp_map_.acquire_block(block_id);
	if (buffer) {
		return buffer;
	}
	{
		// std::cerr << "block_id: " << block_id << ", offset: " << offset << ", size: " << size << std::endl;
		// std::lock_guard<std::mutex> lock(mutex_);
		bool found = free_buffers_.try_dequeue(buffer);
		// std::cerr << "dequeue: " << found << std::endl;
		if (!found) {
			for (int i = 0; i < retry; i++) {
				lp_map_.recycle(free_buffers_);
				found = free_buffers_.try_dequeue(buffer);
				// std::cerr << "dequeue: " << i << std::endl;
				if (found) {
					break;
				}
			}
		}
		if (!found) {
			std::cerr << "Failed to get free buffer " << std::endl;
			return nullptr;
		}
	}

	ssize_t read_bytes = pread(fd_, buffer, size, offset);
	if (read_bytes != static_cast<ssize_t>(size)) {
		std::cerr << "Failed to read file at offset " << offset << std::endl;
		exit(-1);
	}
	char *placed_buffer = nullptr;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		placed_buffer = lp_map_.set_block_acquired(block_id, buffer);
	}
	if (placed_buffer != buffer) {
		// another thread has set the block
		free_buffers_.try_enqueue(buffer);
	}
	return placed_buffer;
}

int VecBufferPool::get_meta(size_t offset, size_t length, char *buffer) {
	ssize_t read_bytes = pread(fd_, buffer, length, offset);
	if (read_bytes != static_cast<ssize_t>(length)) {
		std::cerr << "Failed to read file at offset " << offset << std::endl;
		exit(-1);
	}
	return 0;
}

char* VecBufferPoolHandle::get_block(size_t offset, size_t size, size_t block_id) {
	char *buffer = pool.acquire_buffer(block_id, offset, size, 5);
	return buffer;
}

int VecBufferPoolHandle::get_meta(size_t offset, size_t length, char *buffer) {
	return pool.get_meta(offset, length, buffer);
}

void VecBufferPoolHandle::release_one(block_id_t block_id) {
	pool.lp_map_.release_block(block_id);
}

void VecBufferPoolHandle::acquire_one(block_id_t block_id) {
	pool.lp_map_.acquire_block(block_id);
}

}  // namespace ailego
}  // namespace zvec