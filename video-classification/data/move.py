import glob
import os
import os.path
def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    filename = parts[2]
    filename_no_ext = filename.split('.')[0]
    classname = parts[1]
    train_or_test = parts[0]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))
def move_tain():
    data_file = []
    folders = ['train', 'test']
    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))
        for fir_class in class_folders:
            sec_class = glob.glob(os.path.join(fir_class, '*'))
            for vid_class in sec_class:
                class_files = glob.glob(os.path.join(vid_class, '*.mp4'))
                for video_path in class_files:
                    # Get the parts of the file.
                    video_parts = get_video_parts(video_path)
                    print(video_parts)

                    train_or_test, classname, filename_no_ext, filename = video_parts

                    # Only extract if we haven't done it yet. Otherwise, just get
                    # the info.
                    if not check_already_extracted(video_parts):
                        # Now extract it

                        dest = os.path.join(train_or_test, classname,
                                            filename_no_ext + '.mp4')
                        print(dest)
                        os.rename(video_path, dest)
                        print('success')
def move_testfromtrain():
    data_file = []
    folders = ['train']
    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))
        for fir_class in class_folders:
            class_files = glob.glob(os.path.join(fir_class, '*.mp4'))
            test_num=0
            move_num=len(class_files)/4
            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)
                print(video_parts)
                train_or_test, classname, filename_no_ext, filename = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                if not check_already_extracted(video_parts):
                    # Now extract it

                    dest = os.path.join('test', classname,
                                        filename_no_ext + '.mp4')
                    if not os.path.exists(os.path.join('test', classname)):
                        os.mkdir(os.path.join('test', classname))
                    print(dest)
                    os.rename(video_path, dest)
                    print('success')
                    test_num+=1
                    if test_num > move_num:
                        test_num=0
                        break

def deleteimage():
    folders = ['train', 'test']
    for folder in folders:
        class_folders = glob.glob(os.path.join(folder, '*'))
        for fir_class in class_folders:
            class_files = glob.glob(os.path.join(fir_class, '*.jpg'))
            for img_path in class_files:
                os.remove(img_path)
move_testfromtrain()