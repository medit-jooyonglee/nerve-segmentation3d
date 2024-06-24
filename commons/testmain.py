

def main():
    print('hello')

if __name__=='__main__':
    main()

import pyinstaller_versionfile

pyinstaller_versionfile.create_versionfile(
    output_file="versionfile.txt",
    version="1.2.3.4",
    company_name="My Imaginary Company",
    file_description="Simple App",
    internal_name="Simple App",
    legal_copyright="© My Imaginary Company. All rights reserved.",
    original_filename="SimpleApp.exe",
    product_name="Simple App",
    translations=[0, 1200]
)