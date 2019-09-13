"""
This file will serve as an interface of the API for users.
TODO: We need to define the format of the database so that adapters for each dataset can be created
@Jaeho Bang

"""

import abc #abstract class package


class Interface(abc.ABC):


  @abc.abstractmethod
  def save_video(self, video_name, *options):
    """

    :param video_name: name of the video
    :param options:
          This parameter will need to pack a lot of information, here are examples of what is needed
          1. whether the video is compressed or not
          2. if not compressed, give me the frames
          3. if compressed, give me the location to the file
          4. whether there are annotations
          5. if annotations, give me the annotations in a pandas table format
    :return:
    """
    pass

  @abc.abstractmethod
  def load_video(self, video_name, *options):
    """

    :param filename: name of the video
    :param options:
        1. Whether you want the compressed format
        2. Whether you want it in the uncompressed format
        3. Whether you want just the frames
        4. Whether you want all the annotations (this is going to be a all or none approach)
    :return:
    """
    pass


