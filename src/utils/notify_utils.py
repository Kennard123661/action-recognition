
import os
import requests
import socket
import inspect

BOT_TOKEN = '1030562588:AAGNGH_WLFUOJ91-nbHDRFxRrd8e5aRF5aY'
CHAT_ID = 375385701


def send_telegram_notification(message, msg_level='INFO'):
    assert msg_level in ['INFO', 'ERROR']
    payload = {
        'chat_id': CHAT_ID,
        'text': msg_level + ': ' + message + '\nPC name: {}'.format(socket.gethostname()),
        'parse_mode': 'HTML'
    }
    response = requests.post('https://api.telegram.org/bot{}/sendMessage'.format(BOT_TOKEN), data=payload)
    return response.status_code == 200


def telegram_watch(function):
    def wrapper():
        source_file = inspect.getsourcefile(function)
        source_file = os.path.split(source_file)[-1].split('.')[:-1]
        source_file.append(str(function.__name__))
        source_file = '.'.join(source_file)
        source_file = str(source_file)

        send_telegram_notification(message='task {' + source_file + '} started!')
        try:
            function()
            send_telegram_notification(message='task ' + source_file + ' completed!')
        except:  # catch all errors.
            send_telegram_notification(message='task ' + source_file + ' crashed!', msg_level='ERROR')
            raise

    return wrapper


@telegram_watch
def nothing():
    raise NotImplementedError


if __name__ == '__main__':
    # print(send_telegram_notification('hello'))
    nothing()
