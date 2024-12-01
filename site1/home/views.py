from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np

# Create your views here.
def get_home(request):
    return render(request, 'home.html')

def approximation(request):
    if request.method == 'POST':
                # Đọc file Excel
        file = request.FILES['excel_file']
        data = pd.read_excel(file)  # Đọc file Excel bằng pandas

        # Lấy tên cột do người dùng nhập
        condition_columns_input = request.POST.get('condition_columns')  # Ví dụ: "col1, col2, col3"
        decision_column = request.POST.get('decision_column')  # Tên cột quyết định
        
        # Chuyển tên cột điều kiện thành danh sách
        condition_columns = [col.strip() for col in condition_columns_input.split(',')]
        
        
        # Kiểm tra tính hợp lệ của tên cột
        missing_columns = [col for col in condition_columns + [decision_column] if col not in data.columns]
        if missing_columns:
            error_message = f"Các cột sau không tồn tại trong file Excel: {', '.join(missing_columns)}"
            return render(request, 'approximation.html', {'error': error_message})
        
        # Hàm xử lý xấp xỉ
        def calculate_approximation(data, condition_columns, decision_column):
            grouped_data = data.groupby(decision_column)
            b_lower = []
            b_upper = []
            
            decision_values = data[decision_column].unique()
            for value in decision_values:
                group = grouped_data.get_group(value)
                
                # Tính toán tập B-lower
                lower_set = data[data[condition_columns].apply(tuple, axis=1).isin(group[condition_columns].apply(tuple, axis=1))]
                # Tính toán tập B-upper
                upper_set = data[data[condition_columns].apply(lambda row: any(tuple(row) in group[condition_columns].apply(tuple, axis=1) for group in grouped_data))]
                
                b_lower.append(lower_set)
                b_upper.append(upper_set)
            
            b_lower = pd.concat(b_lower).drop_duplicates()
            b_upper = pd.concat(b_upper).drop_duplicates()
            return b_lower, b_upper
        
        # Tính B-lower, B-upper
        b_lower, b_upper = calculate_approximation(data, condition_columns, decision_column)
        boundary_region = b_upper[~b_upper.index.isin(b_lower.index)]
        
        # Tính độ chính xác
        if len(b_upper) > 0:  # Tránh chia cho 0
            accuracy = len(b_lower) / len(b_upper)
        else:
            accuracy = 0  # Nếu tập B-upper rỗng
        
        # Xuất kết quả
        result = {
            'b_lower': b_lower.to_dict(orient='records'),
            'b_upper': b_upper.to_dict(orient='records'),
            'boundary_region': boundary_region.to_dict(orient='records'),
            'accuracy': accuracy
        }
        return render(request, 'approximation.html', {'result': result})
    
    return render(request, 'approximation.html')



