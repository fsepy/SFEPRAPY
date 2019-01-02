

def list_all_files(root_dir):
    import os, copy

    all_filenames = []

    for dirpath, _, filenames in os.walk(root_dir):

        for f in filenames:

            yield os.path.abspath(os.path.join(dirpath, f))

    # print(all_filenames)


if __name__ == '__main__':
    a = list_all_files(r'D:\Dropbox (OFR-UK)\Bicester_team_projects\Enquiries')

    for i in a:
        if len(i) > 200-43:
            print(i)
