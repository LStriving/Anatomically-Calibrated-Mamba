'''
json
[
  {
    "id": "example_video",
    "path": "D:/data/swallow/example_video.avi",
    "split": "test",
    "duration": 32.0
  }
]
'''
import os
import json

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    if not os.path.exists(os.path.dirname(os.path.abspath(path))):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    

if __name__ == '__main__':
    data = read_json('../anno/swallow_singlestage_without_all.json')['database']
    test_data = {k: v for k, v in data.items() if v['subset'] == 'Test'}
    res = []
    for k, v in test_data.items():
        res.append(
            {
                'id': k,
                'split': 'test',
                'duration': float(k[-2:])
            }
        )

    print(len(res))
    write_json(res, 'test_videos.json')
    