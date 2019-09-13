import cv2
import numpy as np
import os


from eva_storage.decompressionModule import DecompressionModule


class VideoInputModule:

  def __init__(self):
    self.dc = DecompressionModule()
    self.image_array = None

  def _walk_directory(self, directory):
    # we will assume directory is the top folder that contains various subfolders
    assert (os.path.isdir(directory))

    all_directories = [directory]
    done = False

    while not done:
      curr_dir = all_directories[0]
      if os.path.isdir(curr_dir):
        curr_dir = all_directories.pop(0)
        curr_level = os.listdir(curr_dir)
        curr_level.sort()
        for sub in curr_level:
          all_directories.append(os.path.join(curr_dir, sub))
      elif ('jpg' not in curr_dir) and ('jpeg' not in curr_dir) and ('png' not in curr_dir):
        # not a directory but also not a image file
        # we should never arrive here but added just for debugging purposes
        print("We have a non image file in the directory.. this is not allowed...")
        print(curr_dir)
        all_directories.pop(0)
        continue
      else:
        done = True

    print("Number of files to load:", len(all_directories))
    img_tmp = cv2.imread(all_directories[0], cv2.IMREAD_COLOR)
    height, width, channels = img_tmp.shape

    image_array = np.ndarray(shape=(len(all_directories), int(height), int(width), int(channels)), dtype=np.uint8)
    print(image_array.shape)

    for i in range(len(all_directories)):
      file_name = all_directories[i]
      img = cv2.imread(file_name, cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      image_array[i, :, :, :] = img

    print("Done!")
    return image_array

  def debug(self):
    from matplotlib import pyplot as plt
    import random
    fig = plt.figure(figsize=(10, 20))
    columns = 2
    rows = 10
    for idx in range(0, rows):
      fig.add_subplot(rows, columns, idx + 1)
      random_index = random.randint(0, len(self.image_array))
      original = np.copy(self.image_array[random_index])

      plt.imshow(original)

    plt.show()

  def get_image_array(self):
    return self.image_array

  def _convert_video_format(self, filename):
    if os.path.isfile(filename):
      print("given directory is a file..")
      self.image_array = self.dc.convert2images(filename)
    elif os.path.isdir(filename):
      print("given directory is a folder...")
      self.image_array = self._walk_directory(filename)

    # TODO: Do some input checking
    # Make sure the array is uint8
    # Make sure the loaded image array is the same size as the asserted width / height

  def convert_video(self, filename):
    # filename should be file name if compressed video, if not it should be folder name
    self._convert_video_format(filename)



if __name__ == "__main__":
  eva_dir = os.path.abspath('../../')
  data_dir = os.path.join(eva_dir, 'data', 'videos')
  detrac_dir = os.path.join(eva_dir, 'data', 'ua_detrac', 'small-data')
  vim_ = VideoInputModule()
  vim_.convert_video(detrac_dir)
  tmp = vim_.get_image_array()
  print(tmp.shape)

  vim_.debug()
  image_table = vim_.get_image_array()
  print(image_table.shape)
