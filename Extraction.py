import shutil
import urllib3
import xlrd


def downloadFileFromExcel():
    workbook=xlrd.open_workbook("sample.xlsx")
    sheet=workbook.sheet_by_name("in")
    http = urllib3.PoolManager()
    i = 2
    while (i < 2001):
         url=sheet.cell(i, 0).value
         path = 'C:/Users/anpanthri/PycharmProjects/DocumentClassifier/images/test_set/advance_charge_test/advance_charge_test%s.tif' % i
         with http.request('GET', url, preload_content=False) as r, open(path, 'wb') as out_file:
             shutil.copyfileobj(r, out_file)
         i = i + 1;
    while (i < 10002):
        url = sheet.cell(i, 0).value
        path = 'C:/Users/anpanthri/PycharmProjects/DocumentClassifier/images/training_set/advance_charge_train/advance_charge_train%s.tif' % i
        with http.request('GET', url, preload_content=False) as r, open(path, 'wb') as out_file:
            shutil.copyfileobj(r, out_file)
        i = i + 1;

#this error is intentional

downloadFileFromExcel()
