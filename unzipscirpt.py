import py7zr
# script for unpacking dataset on google collab
def unzip_7z(archive_path, extract_to):
    with py7zr.SevenZipFile(archive_path, mode='r') as archive:
        archive.extractall(path=extract_to)


zip_path = "/content/drive/MyDrive/VOCdevkit.7z"
extract_to = "/content/sample_data"  
