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

#include "process_helper.h"

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#include <tlhelp32.h>
#else
#include <sys/stat.h>
#include <sys/syscall.h>
#include <execinfo.h>
#include <pthread.h>
#include <unistd.h>
#endif

#include <signal.h>

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

namespace zvec {
namespace ailego {

#if defined(_WIN32) || defined(_WIN64)
uint32_t ProcessHelper::SelfPid(void) {
  return GetCurrentProcessId();
}

uint32_t ProcessHelper::SelfTid(void) {
  return GetCurrentThreadId();
}

uint32_t ProcessHelper::ParentPid(void) {
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) {
    return (uint32_t)-1;
  }

  DWORD pid = GetCurrentProcessId();
  PROCESSENTRY32 pe = {sizeof(pe)};
  for (BOOL ret = Process32First(snapshot, &pe); ret;
       ret = Process32Next(snapshot, &pe)) {
    if (pe.th32ProcessID == pid) {
      CloseHandle(snapshot);
      return pe.th32ParentProcessID;
    }
  }
  CloseHandle(snapshot);
  return (uint32_t)-1;
}

size_t ProcessHelper::BackTrace(void **buf, size_t size) {
  return RtlCaptureStackBackTrace(1, (DWORD)size, buf, nullptr);
}

bool ProcessHelper::IsExist(uint32_t pid) {
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snapshot == INVALID_HANDLE_VALUE) {
    return false;
  }

  PROCESSENTRY32 pe = {sizeof(pe)};
  for (BOOL ret = Process32First(snapshot, &pe); ret;
       ret = Process32Next(snapshot, &pe)) {
    if (pe.th32ProcessID == pid) {
      CloseHandle(snapshot);
      return true;
    }
  }
  CloseHandle(snapshot);
  return false;
}

void ProcessHelper::Daemon(const char *, const char *) {
  // ::TODO::
}

void ProcessHelper::RegisterSignal(int sig, void f(int)) {
  signal(sig, f);
}

#else  // !_WIN32 && !_WIN64
uint32_t ProcessHelper::SelfPid(void) {
  return getpid();
}

uint32_t ProcessHelper::SelfTid(void) {
#if defined(__linux) || defined(__linux__)
  return syscall(SYS_gettid);
#else
  uint64_t tid = (uint64_t)-1;
  pthread_threadid_np(nullptr, &tid);
  return (uint32_t)tid;
#endif
}

uint32_t ProcessHelper::ParentPid(void) {
  return getppid();
}

size_t ProcessHelper::BackTrace(void **buf, size_t size) {
  size = backtrace(buf, size);

  if (size != 0) {
    --size;
    // Skip the current function
    for (size_t i = 0; i < size; ++i) {
      buf[i] = buf[i + 1];
    }
  }
  return size;
}

bool ProcessHelper::IsExist(uint32_t pid) {
  return (kill(pid, 0) == 0);
}

void ProcessHelper::Daemon(const char *out, const char *err) {
  // Fork off the parent process
  pid_t pid = fork();
  if (pid < 0) {
    exit(EXIT_FAILURE);
  }

  if (pid > 0) {
    // Let the parent terminate
    exit(EXIT_SUCCESS);
  }

  // The child process becomes session leader
  if (setsid() < 0) {
    exit(EXIT_FAILURE);
  }

  // Ignore some signals
  signal(SIGCHLD, SIG_IGN);
  signal(SIGHUP, SIG_IGN);

  // Fork off for the second time
  pid = fork();

  if (pid < 0) {
    exit(EXIT_FAILURE);
  }

  if (pid > 0) {
    // Let the parent terminate
    exit(EXIT_SUCCESS);
  }

  // Set new file permissions
  umask(0);

  // Change the working directory to the root directory
  // or another appropriated directory
  chdir("/");

  // Close all open file descriptors
  for (long x = sysconf(_SC_OPEN_MAX); x >= 0; --x) {
    close(x);
  }

  stdin = nullptr;
  stdout = nullptr;
  stderr = nullptr;

  // Redirect standard output
  if (out) {
    stdout = fopen(out, "w+");
  }

  // Redirect standard error
  if (err) {
    stderr = fopen(err, "w+");
  } else if (out) {
    stderr = fopen(out, "w+");
  }
}

void ProcessHelper::RegisterSignal(int sig, void f(int)) {
  struct sigaction sa;

  sigemptyset(&sa.sa_mask);
  sa.sa_handler = f;
  sa.sa_flags = SA_RESTART;
  sigaction(sig, &sa, nullptr);
}
#endif  // _WIN32 || _WIN64

#if defined(__linux) || defined(__linux__)
static const char *signal_names[32] = {
    "NIL",        // 00 NIL.
    "SIGHUP",     // 01 Hangup (POSIX).
    "SIGINT",     // 02 Interrupt (ANSI).
    "SIGQUIT",    // 03 Quit (POSIX).
    "SIGILL",     // 04 Illegal instruction (ANSI).
    "SIGTRAP",    // 05 Trace trap (POSIX).
    "SIGABRT",    // 06 Abort (ANSI).
    "SIGBUS",     // 07 BUS error (4.2 BSD).
    "SIGFPE",     // 08 Floating-point exception (ANSI).
    "SIGKILL",    // 09 Kill, unblockable (POSIX).
    "SIGUSR1",    // 10 User-defined signal 1 (POSIX).
    "SIGSEGV",    // 11 Segmentation violation (ANSI).
    "SIGUSR2",    // 12 User-defined signal 2 (POSIX).
    "SIGPIPE",    // 13 Broken pipe (POSIX).
    "SIGALRM",    // 14 Alarm clock (POSIX).
    "SIGTERM",    // 15 Termination (ANSI).
    "SIGSTKFLT",  // 16 Stack fault.
    "SIGCHLD",    // 17 Child status has changed (POSIX).
    "SIGCONT",    // 18 Continue (POSIX).
    "SIGSTOP",    // 19 Stop, unblockable (POSIX).
    "SIGTSTP",    // 20 Keyboard stop (POSIX).
    "SIGTTIN",    // 21 Background read from tty (POSIX).
    "SIGTTOU",    // 22 Background write to tty (POSIX).
    "SIGURG",     // 23 Urgent condition on socket (4.2 BSD).
    "SIGXCPU",    // 24 CPU limit exceeded (4.2 BSD).
    "SIGXFSZ",    // 25 File size limit exceeded (4.2 BSD).
    "SIGVTALRM",  // 26 Virtual alarm clock (4.2 BSD).
    "SIGPROF",    // 27 Profiling alarm clock (4.2 BSD).
    "SIGWINCH",   // 28 Window size change (4.3 BSD, Sun).
    "SIGIO",      // 29 I/O now possible (4.2 BSD).
    "SIGPWR",     // 30 Power failure restart (System V).
    "SIGSYS"      // 31 Bad system call.
};

#elif defined(__APPLE__) || defined(__MACH__) || defined(__FreeBSD__) || \
    defined(__NetBSD__) || defined(__OpenBSD__)
static const char *signal_names[33] = {
    "NIL",        // 00 NIL.
    "SIGHUP",     // 01 hangup.
    "SIGINT",     // 02 interrupt.
    "SIGQUIT",    // 03 quit.
    "SIGILL",     // 04 illegal instr. (not reset when caught).
    "SIGTRAP",    // 05 trace trap (not reset when caught).
    "SIGABRT",    // 06 abort().
    "SIGEMT",     // 07 EMT instruction.
    "SIGFPE",     // 08 floating point exception.
    "SIGKILL",    // 09 kill (cannot be caught or ignored).
    "SIGBUS",     // 10 bus error.
    "SIGSEGV",    // 11 segmentation violation.
    "SIGSYS",     // 12 non-existent system call invoked.
    "SIGPIPE",    // 13 write on a pipe with no one to read it.
    "SIGALRM",    // 14 alarm clock.
    "SIGTERM",    // 15 software termination signal from kill.
    "SIGURG",     // 16 urgent condition on IO channel.
    "SIGSTOP",    // 17 sendable stop signal not from tty.
    "SIGTSTP",    // 18 stop signal from tty.
    "SIGCONT",    // 19 continue a stopped process.
    "SIGCHLD",    // 20 to parent on child stop or exit.
    "SIGTTIN",    // 21 to readers pgrp upon background tty read.
    "SIGTTOU",    // 22 like TTIN if (tp->t_local&LTOSTOP).
    "SIGIO",      // 23 input/output possible signal.
    "SIGXCPU",    // 24 exceeded CPU time limit.
    "SIGXFSZ",    // 25 exceeded file size limit.
    "SIGVTALRM",  // 26 virtual time alarm.
    "SIGPROF",    // 27 profiling time alarm.
    "SIGWINCH",   // 28 window size changes.
    "SIGINFO",    // 29 information request.
    "SIGUSR1",    // 30 user defined signal 1.
    "SIGUSR2",    // 31 user defined signal 2.
    "SIGTHR"      // 32 reserved by thread library.
};

#elif defined(_WIN32) || defined(_WIN64)
static const char *signal_names[] = {
    "NIL",       // 00
    "NIL",       // 01
    "SIGINT",    // 02 interrupt
    "NIL",       // 03
    "SIGILLL",   // 04 illegal instruction - invalid function image
    "NIL",       // 05
    "NIL",       // 06
    "NIL",       // 07
    "SIGFPE",    // 08 floating point exception
    "NIL",       // 09
    "NIL",       // 10
    "SIGSEGV",   // 11 segment violation
    "NIL",       // 12
    "NIL",       // 13
    "NIL",       // 14
    "SIGTERM",   // 15 Software termination signal from kill
    "NIL",       // 16
    "NIL",       // 17
    "NIL",       // 18
    "NIL",       // 19
    "NIL",       // 20
    "SIGBREAK",  // 21 Ctrl-Break sequence
    "SIGABRT",   // 22 abnormal termination triggered by abort call
};
#endif

void ProcessHelper::IgnoreSignal(int sig) {
  signal(sig, SIG_IGN);
}

const char *ProcessHelper::SignalName(int sig) {
  if (sig >= 0 && sig < (int)(sizeof(signal_names) / sizeof(signal_names[0]))) {
    return signal_names[sig];
  }
  return signal_names[0];
}

}  // namespace ailego
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif