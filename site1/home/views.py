from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

# Create your views here.
def get_home(request):
    return render(request, 'home.html')

def approximation(request):
    if request.method == 'POST':
        file = request.FILES['excel_file']
        data = pd.read_excel(file)  # Đọc file Excel bằng pandas
        # Xử lý logic xấp xỉ tại đây (ví dụ: tính B-dưới, B-trên, vùng biên)
        result = "Kết quả xấp xỉ sẽ hiển thị ở đây."
        return render(request, 'approximation.html', {'result': result})
    return render(request, 'approximation.html')

