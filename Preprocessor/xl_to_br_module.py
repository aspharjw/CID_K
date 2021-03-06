from xlrd import open_workbook
from bookingreview import BookingReview
import re

def xl_to_BookingReview(file_dir):
    extraction_format_re = '[^ a-zA-Z0-9.!?ㄱ-ㅣ가-힣\n]+'
    kreng = re.compile(extraction_format_re)

    wb = open_workbook(file_dir)
    for sheet in wb.sheets():
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols

        BRList = []

        count = 0
        rows = []
        # 모든 데이터 처리를 원할 경우 -> range(1, number_of_rows)

        for row in range(1, number_of_rows):
            if(count % 10000 == 0):
                print("processing " + str(count) + " th unit...")
            count += 1

            values = []
            values.append(kreng.sub('',str(sheet.cell(row,0).value)))
            values.append(str(sheet.cell(row,1).value))
            values.append(int(sheet.cell(row,2).value))
            values.append(kreng.sub('',str(sheet.cell(row,3).value)))
            values.append(float(sheet.cell(row,5).value))
            values.append(bool(sheet.cell(row,7).value))
            values.append(int(sheet.cell(row,6).value))

            item = BookingReview(*values)
            BRList.append(item)

    return BRList
