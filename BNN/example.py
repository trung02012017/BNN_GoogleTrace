import pandas as pd
import os.path


def saveData(combination, loss_test_act, epoch_i, result_file_path):
    combination_x = [combination]
    result = {'combination': combination_x,
              'loss': loss_test_act,
              'epoch': epoch_i}

    df = pd.DataFrame(result)
    if not os.path.exists(result_file_path):
        columns = ['combination', 'loss', 'epoch']
        df[columns]
        df.to_csv('result_encoder_decoder.csv', index=False, columns=columns)
    else:
        with open('result_encoder_decoder.csv', 'a') as csv_file:
            df.to_csv(csv_file,  mode='a', header=False, index=False)

    name = ''
    name += str(combination)
    name += ' epoch='
    name += str(epoch_i)
    name += ' loss='
    name += str(loss_test_act)
    name += '.png'
    print(name)


saveData([1, 2, 3], 1, 1, "result_encoder_decoder.csv")