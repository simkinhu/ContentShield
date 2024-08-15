import pkg_resources
import subprocess
import sys


def generate_requirements():
    # 获取当前环境中安装的所有包
    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])

    # 将包列表写入 requirements.txt 文件
    with open('requirements.txt', 'w') as f:
        for item in installed_packages_list:
            f.write(f"{item}\n")

    print("requirements.txt 文件已生成。")


def generate_requirements_freeze():
    # 使用 pip freeze 命令生成 requirements.txt
    with open('requirements.txt', 'w') as f:
        subprocess.call([sys.executable, "-m", "pip", "freeze"], stdout=f)

    print("requirements.txt 文件已通过 pip freeze 生成。")


if __name__ == "__main__":
    # 使用 pkg_resources 方法
    generate_requirements()

    # 或者使用 pip freeze 方法
    # generate_requirements_freeze()