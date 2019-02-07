import json

if __name__ == '__main__':
    
    with open('runtime_metric_obj.json', 'r') as f:
        data = [json.loads(line) for line in f]
        print data
