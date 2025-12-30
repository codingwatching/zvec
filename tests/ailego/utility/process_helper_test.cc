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

#include <signal.h>
#include <iostream>
#include <ailego/utility/process_helper.h>
#include <gtest/gtest.h>

using namespace zvec::ailego;

TEST(ProcessHelper, General) {
  EXPECT_NE(0u, ProcessHelper::SelfPid());
  EXPECT_TRUE(ProcessHelper::IsExist(ProcessHelper::SelfPid()));

  EXPECT_EQ("SIGABRT", std::string(ProcessHelper::SignalName(SIGABRT)));
  EXPECT_EQ("NIL", std::string(ProcessHelper::SignalName(255)));

  ProcessHelper::IgnoreSignal(SIGINT);
  ProcessHelper::IgnoreSignal(SIGTERM);

  void *buf[128];
  EXPECT_NE(0u, ProcessHelper::BackTrace(buf, 128));
}
