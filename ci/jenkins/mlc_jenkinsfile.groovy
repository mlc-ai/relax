#!groovy
// -*- mode: groovy -*-

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// ============================= IMPORTANT NOTE =============================
// To keep things simple
// This file is manually updated to maintain unity branch specific builds.
// Please do not send this file to main


import org.jenkinsci.plugins.pipeline.modeldefinition.Utils


tvm_lib = 'build/libtvm.so, build/libtvm_runtime.so, build/config.cmake'
tvm_lib_cuda = tvm_lib + ", build/libfpA_intB_gemm.so, build/libflash_attn.so"
docker_run = 'docker/bash.sh' // command to start a docker container
max_time = 240 // timeout in minutes

ci_lint    = 'tlcpackstaging/ci_lint:20230504-142417-4d37a0a0'
ci_cuda    = 'tlcpackstaging/ci_gpu:20230504-142417-4d37a0a0'
ci_cpu     = 'tlcpackstaging/ci_cpu:20230513-200357-e54bbc73'
ci_wasm    = 'tlcpack/ci-wasm:v0.72'
ci_i386    = 'tlcpack/ci-i386:v0.75'
ci_qemu    = 'tlcpack/ci-qemu:v0.11'
ci_arm     = 'tlcpack/ci-arm:v0.08'
ci_hexagon = 'tlcpackstaging/ci_hexagon:20230504-142417-4d37a0a0'

def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

def init_git() {
  // initialize source codes
  checkout scm
  // Add more info about job node
  sh (
   script: "echo NODE_NAME=${env.NODE_NAME}",
   label: 'Show executor node info',
  )
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh (script: 'git submodule update --init --recursive -f', label: 'Update git submodules')
    }
  }
}

def cancel_previous_build() {
  // cancel previous build if it is not on main.
  if (env.BRANCH_NAME != 'main') {
    def buildNumber = env.BUILD_NUMBER as int
    // Milestone API allows us to cancel previous build
    // with the same milestone number
    if (buildNumber > 1) milestone(buildNumber - 1)
    milestone(buildNumber)
  }
}

def make(build_path, image, config, threads) {
  timeout(time: max_time, unit: 'MINUTES') {
    sh (
      script: "rm -rf ${build_path}",
      label: 'Clean up',
    )
    sh (
      script: "${docker_run} --cpus 1 ${image} ${config} ${build_path}",
      label: 'Configure',
    )
    sh (
      script: "${docker_run} --cpus 1 ${image} ./tests/scripts/mlc/task_mlc_cmake.sh",
      label: 'CMake',
    )
    sh (
      script: "${docker_run} --cpus ${threads} --env BUILD_THREADS=${threads} ${image} ./tests/scripts/mlc/task_mlc_build.sh",
      label: 'Build',
    )
  }
}

def pack_lib(name, libs) {
  sh """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
  stash includes: libs, name: name
}

def unpack_lib(name, libs) {
  unstash name
  sh """
     echo "Unpacked ${libs} from ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
}

cancel_previous_build()

stage('Approval') {
  input id: '1', message: 'Pending. Please reply "\\test" on GitHub to continue.'
}

stage('Prepare') {
  node('JUNRU-CPU-SMALL') {
    // When something is provided in ci_*_param, use it, otherwise default with ci_*
    sh (script: """
      echo "Docker images being used in this build:"
      echo " ci_lint = ${ci_lint}"
      echo " ci_cpu  = ${ci_cpu}"
      echo " ci_cuda = ${ci_cuda}"
      echo " ci_wasm = ${ci_wasm}"
      echo " ci_i386 = ${ci_i386}"
      echo " ci_qemu = ${ci_qemu}"
      echo " ci_arm  = ${ci_arm}"
      echo " ci_hexagon  = ${ci_hexagon}"
    """, label: 'Docker image names')
  }
}

stage('Lint') {
  parallel 'Misc': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/misc')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} tests/lint/check_asf_header.sh --local",
      label: "ASF Header",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/lint/whitespace.sh",
      label: "Whitespace",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} python3 tests/lint/check_file_type.py",
      label: "File Type",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} python3 tests/lint/check_cmake_options.py",
      label: "CMake LibInfo",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/lint/blocklint.sh",
      label: "Non-inclusive Language",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/lint/jnilint.sh",
      label: "JNI",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/lint/rust_format.sh",
      label: "Rust Format",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/lint/docker-format.sh",
      label: "Docker Format",
    )
  }}},
  'Black': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/black')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 2 --env BLACK_THREADS=2 ${ci_lint} ./tests/lint/git-black.sh",
      label: "Black",
    )
  }}},
  'Mypy': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/mypy')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/scripts/task_mypy.sh",
      label: "Mypy",
    )
  }}},
  'Pylint': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/pylint')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 4 --env PYLINT_THREADS=4 ${ci_lint} ./tests/lint/pylint.sh",
      label: "Pylint",
    )
  }}},
  'Flake8': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/flake8')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 4 --env FLAKE8_THREADS=4 ${ci_lint} ./tests/lint/flake8.sh",
      label: "Flake8",
    )
  }}},
  'C++ Docs': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/cppdocs')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 4 ${ci_lint} ./tests/lint/cppdocs.sh",
      label: "Cppdocs",
    )
  }}},
  'Cpplint': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/cpplint')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/lint/cpplint.sh",
      label: "Cpplint",
    )
  }}},
  'Clang Format': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/lint/clang-format')) {
    init_git()
    sh (
      script: "${docker_run} --cpus 1 ${ci_lint} ./tests/lint/git-clang-format.sh",
      label: "Clang Format",
    )
  }}}
}

stage('Build') {
  parallel 'CPU': { node('JUNRU-CPU-LARGE') { ws(per_exec_ws('tvm/build/cpu')) {
    init_git()
    make('build', "${ci_cpu}", './tests/scripts/task_config_build_cpu.sh', 8)
    pack_lib('cpu', tvm_lib)
  }}},
  'CUDA': { node('JUNRU-CPU-LARGE') { ws(per_exec_ws('tvm/build/cuda')) {
    init_git()
    make('build', "${ci_cuda}", './tests/scripts/task_config_build_gpu.sh', 24)
    sh (
      script: """
if [ -f ./build/3rdparty/libflash_attn/src/libflash_attn.so ]; then
  mv ./build/3rdparty/libflash_attn/src/libflash_attn.so ./build/
fi
""",
      label: "Move libflash_attn.so",
    )
    sh (
      script: """
if [ -f ./build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/libfpA_intB_gemm.so ]; then
  mv ./build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels/libfpA_intB_gemm.so ./build/
fi
""",
      label: "Move libfpA_intB_gemm.so",
    )
    pack_lib('cuda', tvm_lib_cuda)
  }}}
}

stage('Unittest') {
  parallel 'Relax Core IR': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-core-ir')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-ir",
      label: "Relax IR",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-op",
      label: "Relax Operators",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-pass",
      label: "Relax Pass",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-nn-module",
      label: "Relax nn.Module",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-dist",
      label: "Relax DistIR",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-pattern-lang",
      label: "Relax Pattern Lang",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-training",
      label: "Relax Training",
    )
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-tvmscript",
      label: "Relax TVMScript",
    )
  }}},

  'Relax Runtime': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-vm')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-vm",
      label: "Relax VM",
    )
  }}},

  'Relax CUDA': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-cuda')) {
    init_git()
    unpack_lib("cuda", tvm_lib_cuda)
    sh (
      script: "${docker_run} --cpus 8 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-cuda",
      label: "Relax CUDA",
    )
  }}},

  'Relax Integration': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-meta-schedule')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 8 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-meta-schedule",
      label: "Relax MetaSchedule",
    )
    sh (
      script: "${docker_run} --cpus 8 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-relay",
      label: "Relax Relay",
    )
  }}},

  'Relax DNNL': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-dnnl')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-dnnl",
      label: "Relax DNNL",
    )
  }}},

  'Relax ONNX': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-onnx')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-onnx",
      label: "Relax ONNX",
    )
  }}},

  'Relax StableHLO': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-stablehlo')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-stablehlo",
      label: "Relax StableHLO",
    )
  }}},

  'Relax Torch': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/relax-torch')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh relax-torch",
      label: "Relax Torch",
    )
  }}},

  'Arith': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/arith')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh arith",
      label: "Arith",
    )
  }}},

  // 'Disco': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/unittest/disco')) {
  //   init_git()
  //   unpack_lib("cpu", tvm_lib)
  //   sh (
  //     script: "${docker_run} --cpus 1 ${ci_cpu} ./tests/scripts/unity/task_python_relax.sh disco",
  //     label: "Disco",
  //   )
  // }}},

  'Dlight': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/dlight')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh dlight",
      label: "Dlight",
    )
  }}},

  // 'AutoScheduler': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/unittest/auto_scheduler')) {
  //   init_git()
  //   unpack_lib("cpu", tvm_lib)
  //   sh (
  //     script: "${docker_run} --cpus 1 ${ci_cpu} ./tests/scripts/unity/task_python_relax.sh auto_scheduler",
  //     label: "AutoScheduler",
  //   )
  // }}},

  // 'AutoTVM': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/unittest/autotvm')) {
  //   init_git()
  //   unpack_lib("cpu", tvm_lib)
  //   sh (
  //     script: "${docker_run} --cpus 1 ${ci_cpu} ./tests/scripts/unity/task_python_relax.sh autotvm",
  //     label: "AutoTVM",
  //   )
  // }}},

  // 'MetaSchedule': { node('JUNRU-CPU-SMALL') { ws(per_exec_ws('tvm/unittest/meta_schedule')) {
  //   init_git()
  //   unpack_lib("cpu", tvm_lib)
  //   sh (
  //     script: "${docker_run} --cpus 1 ${ci_cpu} ./tests/scripts/unity/task_python_relax.sh meta_schedule",
  //     label: "MetaSchedule",
  //   )
  // }}},

  'Codegen': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/codegen')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh codegen",
      label: "Codegen",
    )
  }}},

  'Uncategorized': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/uncategorized')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh unittest",
      label: "Uncategorized",
    )
  }}},


  'TIR TVMScript': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/tir-tvmscript')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh tir-tvmscript",
      label: "TIR TVMScript",
    )
  }}},

  'TIR Pass': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/tir-pass')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh tir-pass",
      label: "TIR Pass",
    )
  }}},

  'TIR Schedule': { node('JUNRU-GPU') { ws(per_exec_ws('tvm/unittest/tir-schedule')) {
    init_git()
    unpack_lib("cuda", tvm_lib)
    sh (
      script: "${docker_run} --cpus 1 ${ci_cuda} ./tests/scripts/unity/task_python_relax.sh tir-schedule",
      label: "TIR Schedule",
    )
  }}}
}
