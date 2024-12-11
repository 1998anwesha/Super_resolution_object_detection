from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['dota/train/images',
                                     'dota/val/images'],
                      test_folders=['dota/test/images'],
                      min_size=100,
                      output_folder='jsons')
