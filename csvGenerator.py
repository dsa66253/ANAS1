import csv, json, os
from data.config import folder
class csvGenerator():
    def __init__(self, expName, path) -> None:
        self.path = path
        self.expName = expName

    def appendRow(self, header:list, rows:list):
        with open(self.path, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            # write the data
            writer.writerows(rows)
if __name__=="__main__":
    print("helo")
    fields = ["name", "area", "country_code2", "country_code3"]
    rows = [
        {'name': 'Albania',
        'area': 28748,
        'country_code2': 'AL',
        'country_code3': 'ALB'},
        {'name': 'Algeria',
        'area': 2381741,
        'country_code2': 'DZ',
        'country_code3': 'DZA'},
        {'name': 'American Samoa',
        'area': 199,
        'country_code2': 'AS',
        'country_code3': 'ASM'}
    ]
    kth = 0
    filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
    f = open(filePath)
    archDict = json.load(f)
    csvG = csvGenerator("test", "./countries.csv")
    csvG.appendRow(archDict.keys(), [archDict])
    
