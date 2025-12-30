// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pid_file.h"
#include <ailego/utility/process_helper.h>

namespace zvec {
namespace ailego {

bool PidFile::open(const char *path) {
  if (file_.is_valid() || !path || !(*path)) {
    return false;
  }

  if (!File::IsExist(path)) {
    if (!file_.create(path, 0, false)) {
      return false;
    }
    if (!FileLock::TryLock(file_.native_handle())) {
      file_.close();
      return false;
    }
  } else {
    if (!file_.open(path, false, false)) {
      return false;
    }
    if (!FileLock::TryLock(file_.native_handle())) {
      file_.close();
      return false;
    }
    file_.truncate(0);
  }

  std::string str(std::to_string(ProcessHelper::SelfPid()));
  file_.write(str.data(), str.size());
  return true;
}

void PidFile::close(void) {
  if (file_.is_valid()) {
    FileLock::Unlock(file_.native_handle());

    std::string path;
    FileHelper::GetFilePath(file_.native_handle(), &path);
    file_.close();

    if (!path.empty()) {
      File::Delete(path);
    }
  }
}

}  // namespace ailego
}  // namespace zvec