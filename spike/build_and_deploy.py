import os
import sys
import subprocess
import shutil
import re
from pathlib import Path
from typing import Set, List, Optional
import argparse

class DependencyAnalyzer:
    def __init__(self):
        self.system_dlls = {
            'kernel32.dll', 'user32.dll', 'gdi32.dll', 'winspool.drv',
            'comdlg32.dll', 'advapi32.dll', 'shell32.dll', 'ole32.dll',
            'oleaut32.dll', 'uuid.dll', 'odbc32.dll', 'odbccp32.dll',
            'msvcrt.dll', 'ntdll.dll', 'ws2_32.dll', 'wsock32.dll',
            'mswsock.dll', 'rpcrt4.dll', 'secur32.dll', 'netapi32.dll',
            'winmm.dll', 'version.dll', 'shlwapi.dll', 'comctl32.dll',
            'imm32.dll', 'setupapi.dll', 'cfgmgr32.dll', 'devobj.dll',
            'powrprof.dll', 'uxtheme.dll', 'dwmapi.dll', 'winhttp.dll',
            'crypt32.dll', 'wintrust.dll', 'imagehlp.dll', 'psapi.dll',
            'bcrypt.dll', 'ncrypt.dll', 'api-ms-win'
        }
        
        # Add common third-party library paths
        self.third_party_paths = self._get_third_party_paths()
        
    def _get_third_party_paths(self) -> List[Path]:
        """Get common third-party library installation paths"""
        paths = []
        
        # CUDA paths
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        ]
        
        # TensorRT paths
        tensorrt_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\lib",
            r"C:\TensorRT\lib",
            r"C:\TensorRT\lib\x64",
        ]
        
        # FFmpeg paths
        ffmpeg_paths = [
            r"C:\games\ffmpeg-7.1.1\bin",
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
        ]
        
        # cuDNN paths (often in CUDA directory)
        cudnn_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
            r"C:\tools\cuda\bin",
            r"C:\cudnn\bin",
        ]
        
        # Add paths that exist
        for path_list in [cuda_paths, tensorrt_paths, ffmpeg_paths, cudnn_paths]:
            for path_str in path_list:
                path = Path(path_str)
                if path.exists() and path not in paths:
                    paths.append(path)
        
        # Add current PATH directories
        if 'PATH' in os.environ:
            for path_str in os.environ['PATH'].split(os.pathsep):
                if path_str:
                    path = Path(path_str)
                    if path.exists() and path not in paths:
                        paths.append(path)
        
        print(f"Third-party search paths: {[str(p) for p in paths]}")
        return paths
        
    def find_visual_studio_tools(self) -> Optional[Path]:
        """Find Visual Studio tools directory"""
        possible_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC",
        ]
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                # Find the latest version
                versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if versions:
                    latest_version = sorted(versions)[-1]
                    tools_path = Path(base_path) / latest_version / "bin" / "Hostx64" / "x64"
                    if tools_path.exists():
                        return tools_path
        return None
    
    def get_dependencies_dumpbin(self, exe_path: Path) -> Set[str]:
        """Use dumpbin to get dependencies"""
        tools_path = self.find_visual_studio_tools()
        if not tools_path:
            raise RuntimeError("Visual Studio tools not found")
        
        dumpbin_path = tools_path / "dumpbin.exe"
        if not dumpbin_path.exists():
            raise RuntimeError(f"dumpbin.exe not found at {dumpbin_path}")
        
        try:
            result = subprocess.run([
                str(dumpbin_path), "/dependents", str(exe_path)
            ], capture_output=True, text=True, check=True)
            
            dependencies = set()
            lines = result.stdout.split('\n')
            in_section = False
            
            for line in lines:
                line = line.strip()
                if "Image has the following dependencies:" in line:
                    in_section = True
                    continue
                elif in_section:
                    if line == "" or "Summary" in line:
                        break
                    if line.endswith('.dll') or line.endswith('.DLL'):
                        dependencies.add(line.lower())
            
            return dependencies
        except subprocess.CalledProcessError as e:
            print(f"dumpbin failed: {e}")
            return set()
    
    def get_dependencies_pefile(self, exe_path: Path) -> Set[str]:
        """Use pefile library to get dependencies (fallback method)"""
        try:
            import pefile
        except ImportError:
            print("pefile not installed. Install with: pip install pefile")
            return set()
        
        try:
            pe = pefile.PE(str(exe_path))
            dependencies = set()
            
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8').lower()
                    dependencies.add(dll_name)
            
            return dependencies
        except Exception as e:
            print(f"pefile analysis failed: {e}")
            return set()
    
    def find_dll_in_system(self, dll_name: str) -> Optional[Path]:
        """Find DLL in system and third-party paths"""
        all_paths = []
        
        # Add third-party paths first (higher priority)
        all_paths.extend(self.third_party_paths)
        
        # Add system paths
        system_paths = [r'C:\Windows\System32', r'C:\Windows\SysWOW64']
        for path_str in system_paths:
            path = Path(path_str)
            if path.exists():
                all_paths.append(path)
        
        # Search for DLL
        for path in all_paths:
            dll_path = path / dll_name
            if dll_path.exists():
                return dll_path
        
        print(f"Warning: Could not find {dll_name} in any search path")
        return None
    
    def is_system_dll(self, dll_name: str) -> bool:
        """Check if DLL is a system DLL that shouldn't be redistributed"""
        dll_lower = dll_name.lower()
        
        # Don't skip these important DLLs even if they might seem like system DLLs
        important_dlls = {
            'cudart64_', 'cublas64_', 'cublaslt64_', 'curand64_', 'cusparse64_',
            'cusolver64_', 'cufft64_', 'cudnn64_', 'nvinfer', 'nvinfer_plugin',
            'nvonnxparser', 'avcodec', 'avformat', 'avutil', 'swscale', 'swresample'
        }
        
        # Check if this is an important third-party DLL
        for important in important_dlls:
            if important in dll_lower:
                print(f"Important DLL detected: {dll_name}")
                return False
        
        # Check against system DLL patterns
        return any(sys_dll in dll_lower for sys_dll in self.system_dlls)
    
    def analyze_dependencies(self, exe_path: Path, analyzed_files: Set[Path] = None) -> Set[Path]:
        """Analyze dependencies and return paths to DLLs that need to be copied"""
        if analyzed_files is None:
            analyzed_files = set()
        
        # Avoid infinite recursion
        if exe_path in analyzed_files:
            return set()
        
        analyzed_files.add(exe_path)
        print(f"Analyzing dependencies for {exe_path}")
        
        # Try dumpbin first, then pefile as fallback
        deps = self.get_dependencies_dumpbin(exe_path)
        if not deps:
            print("Trying pefile as fallback...")
            deps = self.get_dependencies_pefile(exe_path)
        
        if not deps:
            print("No dependencies found or analysis failed")
            return set()
        
        print(f"Found {len(deps)} dependencies:")
        for dep in sorted(deps):
            print(f"  - {dep}")
        
        # Filter out system DLLs and find actual paths
        dll_paths = set()
        not_found_dlls = []
        
        for dll_name in deps:
            if self.is_system_dll(dll_name):
                print(f"Skipping system DLL: {dll_name}")
                continue
            
            dll_path = self.find_dll_in_system(dll_name)
            if dll_path:
                dll_paths.add(dll_path)
                print(f"Found: {dll_name} -> {dll_path}")
                
                # Recursively analyze this DLL's dependencies
                recursive_deps = self.analyze_dependencies(dll_path, analyzed_files)
                dll_paths.update(recursive_deps)
            else:
                not_found_dlls.append(dll_name)
        
        if not_found_dlls:
            print(f"\nWarning: Could not find the following DLLs:")
            for dll in not_found_dlls:
                print(f"  - {dll}")
            print("These may need to be manually copied or the paths may need to be added.")
        
        return dll_paths

    def find_common_missing_dlls(self, dll_paths: Set[Path]) -> Set[Path]:
        """Find commonly missing DLLs that might not be detected by dependency analysis"""
        additional_dlls = set()
        
        # Check if we have FFmpeg DLLs, if so, add common FFmpeg dependencies
        ffmpeg_dlls = [path for path in dll_paths if any(ffmpeg in path.name.lower() 
                      for ffmpeg in ['avcodec', 'avformat', 'avutil'])]
        
        if ffmpeg_dlls:
            print("FFmpeg DLLs detected, checking for additional FFmpeg dependencies...")
            ffmpeg_dir = ffmpeg_dlls[0].parent
            
            # Common FFmpeg DLLs that might not be directly detected
            common_ffmpeg_dlls = [
                'swresample-5.dll',
                'swscale-8.dll',
                'postproc-58.dll',
                'avfilter-10.dll'
            ]
            
            for dll_name in common_ffmpeg_dlls:
                dll_path = ffmpeg_dir / dll_name
                if dll_path.exists() and dll_path not in dll_paths:
                    print(f"Adding common FFmpeg DLL: {dll_path}")
                    additional_dlls.add(dll_path)
        
        return additional_dlls

class ProjectBuilder:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_dir = project_root / "build"
        
    def clean_build_dir(self):
        """Clean CMake cache and build files"""
        if not self.build_dir.exists():
            return
        
        files_to_remove = [
            "CMakeCache.txt",
            "CMakeFiles",
            "cmake_install.cmake",
            "Makefile"
        ]
        
        print("Cleaning build directory...")
        for item in files_to_remove:
            item_path = self.build_dir / item
            if item_path.exists():
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                    print(f"Removed directory: {item_path}")
                else:
                    item_path.unlink()
                    print(f"Removed file: {item_path}")
        
    def setup_environment(self):
        """Setup Visual Studio environment"""
        vcvarsall_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        ]
        
        for vcvarsall in vcvarsall_paths:
            if os.path.exists(vcvarsall):
                print(f"Setting up Visual Studio environment using {vcvarsall}")
                # Run vcvarsall and capture environment
                cmd = f'"{vcvarsall}" amd64 && set'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse environment variables
                    for line in result.stdout.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                    return True
        
        print("Warning: Could not setup Visual Studio environment")
        return False
    
    def build(self, build_type: str = "Release", clean: bool = True) -> Optional[Path]:
        """Build the project using CMake"""
        print(f"Building project in {build_type} mode...")
        
        # Create build directory
        self.build_dir.mkdir(exist_ok=True)
        
        # Clean build directory if requested
        if clean:
            self.clean_build_dir()
        
        # Configure
        configure_cmd = [
            "cmake",
            "-G", "Visual Studio 17 2022",
            "-A", "x64",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            str(self.project_root)
        ]
        
        print(f"Configuring: {' '.join(configure_cmd)}")
        result = subprocess.run(configure_cmd, cwd=self.build_dir)
        if result.returncode != 0:
            print("CMake configuration failed")
            return None
        
        # Build
        build_cmd = [
            "cmake",
            "--build", ".",
            "--config", build_type
        ]
        
        print(f"Building: {' '.join(build_cmd)}")
        result = subprocess.run(build_cmd, cwd=self.build_dir)
        if result.returncode != 0:
            print("Build failed")
            return None
        
        # Find the executable
        exe_path = self.build_dir / build_type / "spike.exe"
        if not exe_path.exists():
            exe_path = self.build_dir / "spike.exe"
        
        if exe_path.exists():
            print(f"Build successful: {exe_path}")
            return exe_path
        else:
            print("Executable not found after build")
            return None

def main():
    parser = argparse.ArgumentParser(description="Build and deploy exe with dependencies")
    parser.add_argument("--build-type", default="Release", choices=["Debug", "Release"],
                        help="Build type (default: Release)")
    parser.add_argument("--output-dir", type=Path, default=Path("deploy"),
                        help="Output directory for deployment (default: deploy)")
    parser.add_argument("--no-build", action="store_true",
                        help="Skip building, just analyze existing exe")
    parser.add_argument("--exe-path", type=Path,
                        help="Path to existing exe (used with --no-build)")
    parser.add_argument("--no-clean", action="store_true",
                        help="Don't clean build directory before building")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    exe_path = None
    
    if args.no_build:
        if args.exe_path and args.exe_path.exists():
            exe_path = args.exe_path
        else:
            # Look for existing exe
            build_dir = project_root / "build"
            possible_paths = [
                build_dir / args.build_type / "spike.exe",
                build_dir / "spike.exe"
            ]
            for path in possible_paths:
                if path.exists():
                    exe_path = path
                    break
        
        if not exe_path:
            print("No existing exe found. Use --exe-path or remove --no-build")
            return 1
    else:
        # Build the project
        builder = ProjectBuilder(project_root)
        builder.setup_environment()
        exe_path = builder.build(args.build_type, clean=not args.no_clean)
        
        if not exe_path:
            print("Build failed")
            return 1
    
    # Analyze dependencies
    analyzer = DependencyAnalyzer()
    dll_paths = analyzer.analyze_dependencies(exe_path)
    
    # Add commonly missing DLLs
    additional_dlls = analyzer.find_common_missing_dlls(dll_paths)
    dll_paths.update(additional_dlls)
    
    # Create deployment directory
    deploy_dir = args.output_dir
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy executable
    exe_dest = deploy_dir / exe_path.name
    print(f"Copying {exe_path} -> {exe_dest}")
    shutil.copy2(exe_path, exe_dest)
    
    # Copy dependencies
    if dll_paths:
        print(f"\nCopying {len(dll_paths)} dependencies:")
        for dll_path in dll_paths:
            dll_dest = deploy_dir / dll_path.name
            print(f"  {dll_path} -> {dll_dest}")
            shutil.copy2(dll_path, dll_dest)
    else:
        print("No dependencies to copy")
    
    print(f"\nDeployment complete in: {deploy_dir.absolute()}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 