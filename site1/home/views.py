from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from django.contrib import messages
import os

# Create your views here.
def get_home(request):
    return render(request, 'home.html')


def process_excel(file_path, target_set, attributes_set):
    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(file_path)

    # Tính các lớp tương đương
    equivalence_classes = {}
    for _, row in df.iterrows():
        key = tuple(row[attr] for attr in attributes_set)  # Tạo khóa dựa trên thuộc tính
        equivalence_classes.setdefault(key, []).append(row['quyetdinh'])  # Nhóm các quyết định theo khóa

    # Tìm Xấp xỉ dưới và Xấp xỉ trên
    lower_approx = set()
    upper_approx = set()
    for _, objs in equivalence_classes.items():
        obj_set = set(objs)
        if obj_set.issubset(target_set):  # Nếu tất cả các đối tượng thuộc tập X
            lower_approx.update(objs)
        if obj_set.intersection(target_set):  # Nếu có giao với tập X
            upper_approx.update(objs)

    # Tính độ chính xác
    accuracy = len(lower_approx) / len(upper_approx) if upper_approx else 0

    return lower_approx, upper_approx, accuracy

def approximation(request):
    context = {}

    # Lấy đường dẫn file từ session nếu có
    file_path = request.session.get('file_path')
    context['file_path'] = file_path
    context['file_name'] = os.path.basename(file_path) if file_path else None

    if request.method == 'POST':
        if 'excel_file' in request.FILES:  # Form tải file Excel
            excel_file = request.FILES['excel_file']
            
            # Lưu file tạm thời
            file_path = f'tmp/{excel_file.name}'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb+') as destination:
                for chunk in excel_file.chunks():
                    destination.write(chunk)

            # Lưu đường dẫn file vào session
            request.session['file_path'] = file_path
            context['file_name'] = excel_file.name
            context['file_path'] = file_path
            context['message'] = "Tải file thành công! Vui lòng nhập thông tin để tính xấp xỉ."

        elif 'target_set' in request.POST and 'attributes_set' in request.POST:  # Form nhập tập X và tập thuộc tính
            # Lấy dữ liệu từ session
            file_path = request.session.get('file_path')

            if file_path and os.path.exists(file_path):  # Kiểm tra file có tồn tại
                try:
                    # Lấy tập đối tượng và tập thuộc tính từ input
                    target_set = request.POST.get('target_set').split(',')
                    attributes_set = request.POST.get('attributes_set').split(',')

                    # Lưu lại giá trị để hiển thị khi reload
                    context['target_set'] = ','.join(target_set)
                    context['attributes_set'] = ','.join(attributes_set)

                    # Gọi hàm xử lý
                    lower_approx, upper_approx, accuracy = process_excel(file_path, set(target_set), attributes_set)

                    # Đưa kết quả vào context
                    context['lower_approx'] = lower_approx
                    context['upper_approx'] = upper_approx
                    context['accuracy'] = accuracy
                except Exception as e:
                    context['error'] = f"Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
            else:
                context['error'] = "Không tìm thấy file Excel đã tải lên. Vui lòng tải lại file."
    
    return render(request, 'approximation.html', context)