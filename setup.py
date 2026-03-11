from setuptools import setup, find_packages

setup(
    name='lumen',
    version='0.1.0',
    description='LUMEN: Ultra-Lightweight Super-Resolution for STM32 Cortex-M',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'numpy>=1.21.0,<2.0.0',
        'opencv-python>=4.5.0',
        'pillow>=9.0.0',
        'pyyaml>=6.0',
        'tensorboard>=2.9.0',
        'tqdm>=4.64.0',
        'onnx>=1.12.0',
        'scikit-image>=0.19.0',
    ],
)
