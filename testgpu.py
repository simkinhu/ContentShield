import os
import ast
import sys
import importlib

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata


def get_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:  # 非相对导入
                imports.add(node.module.split('.')[0])

    return imports


def get_package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def analyze_dependencies():
    all_imports = set()

    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                all_imports.update(get_imports(file_path))

    # 过滤掉标准库模块
    std_libs = set(sys.stdlib_module_names)
    third_party_imports = all_imports - std_libs

    # 生成 requirements.txt
    with open('requirements.txt', 'w') as f:
        for package in sorted(third_party_imports):
            version = get_package_version(package)
            if version:
                f.write(f"{package}=={version}\n")
            else:
                f.write(f"{package}\n")

    print("requirements.txt 文件已生成，包含项目中实际使用的第三方依赖。")


if __name__ == "__main__":
    analyze_dependencies()