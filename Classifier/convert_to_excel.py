from bookingreview import BookingReview
import xlsxwriter

def convert_to_excel(tuplist, out_link):
    workbook = xlsxwriter.Workbook(out_link)
    worksheet = workbook.add_worksheet()

    first_row = ['company', 'id', 'rating', 'context', 'time float',
                 'index', 'label', 'p(spam)', 'predict']

    row = 0
    col = 0
    for f in first_row:
        worksheet.write(row, col, f)
        col += 1
    col = 0
    row += 1

    for br, p, label in tuplist:
        worksheet.write(row, col, br.company)
        col += 1
        worksheet.write(row, col, br.id)
        col += 1
        worksheet.write(row, col, br.rate)
        col += 1
        worksheet.write(row, col, br.context)
        col += 1
        worksheet.write(row, col, br.post_time)
        col += 1
        worksheet.write(row, col, br.review_id)
        col += 1
        worksheet.write(row, col, int(br.label))
        col += 1
        worksheet.write(row, col, p)
        col += 1
        worksheet.write(row, col, label)
        col = 0
        row += 1
    workbook.close()
