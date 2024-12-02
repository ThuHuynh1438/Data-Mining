from django.shortcuts import redirect, render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from django.contrib import messages
import os

# Create your views here.
def get_home(request):
    return render(request, 'home.html')
def TapPhoBien(request):
    return render(request, 'TapPhoBien.html')

def process_excel(file_path, target_set, attributes_set):
    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(file_path)

    # Xác định các lớp tương đương theo các thuộc tính
    equivalence_classes = {}

    for index, row in df.iterrows():
        key = tuple(row[attr] for attr in attributes_set)
        if key not in equivalence_classes:
            equivalence_classes[key] = set()
        equivalence_classes[key].add(f'o{index+1}')
    
    # Tính toán xấp xỉ dưới và xấp xỉ trên
    lower_approx = set()
    upper_approx = set()

    # Duyệt qua các lớp tương đương để tìm xấp xỉ dưới và trên
    for eq_class in equivalence_classes.values():
        # Kiểm tra xấp xỉ dưới (Lower Approximation): Nếu tất cả các phần tử trong lớp tương đương đều thuộc X
        if eq_class.issubset(target_set):  # Nếu lớp tương đương là con của tập X
            lower_approx.update(eq_class)
        
        # Kiểm tra xấp xỉ trên (Upper Approximation): Nếu ít nhất 1 phần tử trong lớp tương đương thuộc X
        if not eq_class.isdisjoint(target_set):  # Lớp tương đương giao với target_set không rỗng
            upper_approx.update(eq_class)

    # Tính độ chính xác nếu xấp xỉ trên không rỗng
    accuracy = len(lower_approx) / len(upper_approx) if upper_approx else None

    return equivalence_classes, lower_approx, upper_approx, accuracy


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
            request.session['file_name'] = excel_file.name  # Lưu tên file vào session
            context['file_name'] = excel_file.name
            context['file_path'] = file_path
            context['message'] = "Tải file thành công! Vui lòng nhập thông tin để tính xấp xỉ."

        elif 'target_set' in request.POST and 'attributes_set' in request.POST:  # Form nhập tập X và tập thuộc tính
            # Lấy dữ liệu từ session
            file_path = request.session.get('file_path')
            file_name = request.session.get('file_name')  # Lấy tên file từ session
            
            if file_path and file_name and os.path.exists(file_path):  # Kiểm tra file có tồn tại
                try:
                    # Lấy tập đối tượng và tập thuộc tính từ input
                    target_set = request.POST.get('target_set').split(',')
                    attributes_set = request.POST.get('attributes_set').split(',')

                    # Lưu lại giá trị để hiển thị khi reload
                    context['target_set'] = ','.join(target_set)
                    context['attributes_set'] = ','.join(attributes_set)

                    # Gọi hàm xử lý
                    equivalence_classes, lower_approx, upper_approx, accuracy = process_excel(file_path, set(target_set), attributes_set)

                    # Đưa kết quả vào context
                    context['equivalence_classes'] = equivalence_classes  # Đưa lớp tương đương vào context
                    context['lower_approx'] = lower_approx if lower_approx else 'Không có đối tượng phù hợp.'
                    context['upper_approx'] = upper_approx if upper_approx else 'Không có đối tượng phù hợp.'
                    context['accuracy'] = accuracy if accuracy is not None else 'Không thể tính toán độ chính xác.'

                    # Sau khi tính toán xong, xóa dữ liệu trong session để tránh lưu lại thông tin
                    del request.session['file_path']  # Xóa đường dẫn file
                    del request.session['file_name']  # Xóa tên file
                    del request.session['target_set']  # Xóa tập X nếu cần
                    del request.session['attributes_set']  # Xóa tập thuộc tính nếu cần

                except Exception as e:
                    context['error'] = f"Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
            else:
                context['error'] = "Không tìm thấy file Excel đã tải lên. Vui lòng tải lại file."
    
    return render(request, 'approximation.html', context)
