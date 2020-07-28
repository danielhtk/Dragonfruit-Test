import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--not2delete', default='',type=str)
    args = parser.parse_args()
    folders=os.listdir("images/")
    for folder in folders:
        if(os.path.isdir(os.path.join("images/",folder))) and folder != args.not2delete:
            for filename in os.listdir(os.path.join("images/",folder)):
                file_path = os.path.join("images/",os.path.join(folder, filename))
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    main()
