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

#include <chrono>
#include <thread>
#include <ailego/io/pid_file.h>
#include <gtest/gtest.h>

using namespace zvec::ailego;

TEST(PidFile, General) {
  const std::string path = "pid_file_test.pid";

  PidFile pid_file;
  ASSERT_FALSE(pid_file.is_valid());
  ASSERT_TRUE(pid_file.open(path));

  ASSERT_TRUE(pid_file.is_valid());
  ASSERT_TRUE(File::IsExist(path));
  //   std::this_thread::sleep_for(std::chrono::milliseconds(60000));

  pid_file.close();
  ASSERT_FALSE(File::IsExist(path));
}
