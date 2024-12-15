import base64
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
import os
from sklearn.tree import _tree
from django.conf import settings
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# Create your views here.
def get_home(request):
    return render(request, 'home.html')

# Hệ số tương quan
def HSTuongQuan(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            try:
                uploaded_file = request.FILES['file']
                # Đọc file Excel
                data = pd.read_excel(uploaded_file)

                # Kiểm tra dữ liệu đầu vào
                if data.shape[1] < 2:
                    return JsonResponse({'error': 'File phải chứa ít nhất hai cột dữ liệu.'}, status=400)

                # Lấy hai cột đầu tiên
                column_x, column_y = data.columns[:2]
                x = data[column_x].dropna()
                y = data[column_y].dropna()

                # Kiểm tra dữ liệu hợp lệ
                if len(x) == 0 or len(y) == 0:
                    return JsonResponse({'error': 'Dữ liệu trong cột không hợp lệ hoặc rỗng.'}, status=400)

                # Tính toán các thông số
                mean_x = np.mean(x)
                mean_y = np.mean(y)
                variance_x = np.mean(x ** 2) - mean_x ** 2
                variance_y = np.mean(y ** 2) - mean_y ** 2
                phsai_x = np.sqrt(variance_x)
                phsai_y = np.sqrt(variance_y)
                mean_xy = np.mean(x * y)

                b1 = (mean_xy - mean_x * mean_y) / variance_x
                b0 = mean_y - b1 * mean_x
                r = (mean_xy - mean_x * mean_y) / (np.sqrt(variance_x) * np.sqrt(variance_y))

                # Xác định ý nghĩa của hệ số tương quan
                if abs(r) <= 0.1:
                    interpretation = "Mối tương quan quá thấp"
                elif abs(r) <= 0.3:
                    interpretation = "Mối tương quan thấp"
                elif abs(r) <= 0.5:
                    interpretation = "Mối tương quan trung bình"
                elif abs(r) <= 0.7:
                    interpretation = "Mối tương quan cao"
                else:
                    interpretation = "Mối tương quan rất cao"


                # Trả về kết quả dưới dạng JSON
                return JsonResponse({
                    'column_x': column_x,
                    'column_y': column_y,
                    'b1': b1,
                    'b0': b0,
                    'r': r,
                    'mean_x': mean_x,
                    'mean_y': mean_y,
                    'mean_xy': mean_xy,
                    'variance_x': variance_x,
                    'phsai_x': phsai_x,
                    'phsai_y': phsai_y,
                    'correlation': f"Hệ số tương quan (r): {r:.3f}",
                    'interpretation': interpretation
                })
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)

    return render(request, 'HSTuongQuan.html')

# Tập phổ biến
def TapPhoBien(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            try:
                uploaded_file = request.FILES['file']
                data = pd.read_excel(uploaded_file)

                # Kiểm tra nếu file chứa ít nhất 2 cột
                if data.shape[1] < 2:
                    return JsonResponse({'error': 'File phải chứa ít nhất hai cột dữ liệu.'}, status=400)

                # Lấy tên cột từ request (hoặc mặc định lấy cột đầu tiên và thứ hai)
                column1 = request.POST.get('column1', data.columns[0])
                column2 = request.POST.get('column2', data.columns[1])

                # Kiểm tra cột người dùng chọn có tồn tại
                if column1 not in data.columns or column2 not in data.columns:
                    return JsonResponse({'error': f'Cột {column1} hoặc {column2} không tồn tại trong file.'}, status=400)

                # Xây dựng bảng transaction_data
                transaction_data = data.pivot_table(index=column1, columns=column2, aggfunc='size', fill_value=0)
                transaction_data = transaction_data.applymap(lambda x: 1 if x > 0 else 0)

                # Nhận giá trị minsupp và minconf từ request
                minsupp = float(request.POST.get('support', 0.5))
                minconf = float(request.POST.get('confidence', 0.7))
                num_transactions = len(transaction_data)

                # Hàm tìm các tập phổ biến
                def find_frequent_itemsets(transaction_data, minsupp):
                    frequent_itemsets = {}
                    item_support = (transaction_data.sum(axis=0) / num_transactions).to_dict()
                    current_itemsets = {frozenset([item]): support for item, support in item_support.items() if support >= minsupp}
                    frequent_itemsets.update(current_itemsets)

                    k = 2
                    while current_itemsets:
                        new_combinations = list(combinations(set().union(*current_itemsets.keys()), k))
                        itemset_counts = {frozenset(itemset): (transaction_data[list(itemset)].all(axis=1).sum()) for itemset in new_combinations}
                        current_itemsets = {itemset: count / num_transactions for itemset, count in itemset_counts.items() if count / num_transactions >= minsupp}
                        frequent_itemsets.update(current_itemsets)
                        k += 1

                    return frequent_itemsets

                # Hàm tìm tập cực đại
                def find_maximal_itemsets(frequent_itemsets):
                    maximal_itemsets = []
                    for itemset in frequent_itemsets:
                        is_maximal = True
                        for other_itemset in frequent_itemsets:
                            if itemset != other_itemset and itemset.issubset(other_itemset):
                                is_maximal = False
                                break
                        if is_maximal:
                            maximal_itemsets.append(itemset)
                    return maximal_itemsets

                # Hàm tạo luật kết hợp
                def generate_association_rules(frequent_itemsets, minconf):
                    rules = []
                    for itemset, support in frequent_itemsets.items():
                        if len(itemset) > 1:
                            for consequence in itemset:
                                antecedent = itemset - frozenset([consequence])
                                if antecedent:
                                    antecedent_support = frequent_itemsets[antecedent]
                                    confidence = support / antecedent_support
                                    if confidence >= minconf:
                                        rules.append({
                                            'antecedent': list(antecedent),
                                            'consequence': [consequence],
                                            'confidence': confidence
                                        })
                    return rules

                # Tìm các tập phổ biến, cực đại và luật kết hợp
                frequent_itemsets = find_frequent_itemsets(transaction_data, minsupp)
                maximal_itemsets = find_maximal_itemsets(frequent_itemsets)
                association_rules = generate_association_rules(frequent_itemsets, minconf)

                # Trả về kết quả dưới dạng JSON
                return JsonResponse({
                    'frequent_itemsets': [{'itemset': list(itemset), 'support': support} for itemset, support in frequent_itemsets.items()],
                    'maximal_itemsets': [list(itemset) for itemset in maximal_itemsets],
                    'association_rules': association_rules,
                })
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)

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

    # Hiển thị preview dữ liệu nếu đã tải file lên
    if file_path and os.path.exists(file_path):
        try:
            # Xác định định dạng file
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.csv':
                data = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Định dạng file không được hỗ trợ.")

            # Tạo bảng preview
            data_preview = data.head(10).to_html(classes='table table-striped')  # Hiển thị 10 dòng đầu
            context['data_preview'] = data_preview  # Thêm vào context

        except Exception as e:
            context['error'] = f"Không thể hiển thị dữ liệu: {str(e)}"

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

            # Hiển thị preview dữ liệu
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == '.csv':
                    data = pd.read_csv(file_path)
                elif file_extension in ['.xls', '.xlsx']:
                    data = pd.read_excel(file_path)
                else:
                    raise ValueError("Định dạng file không được hỗ trợ.")

                data_preview = data.head(10).to_html(classes='table table-striped')
                context['data_preview'] = data_preview

            except Exception as e:
                context['error'] = f"Không thể hiển thị dữ liệu: {str(e)}"

        elif 'target_set' in request.POST and 'attributes_set' in request.POST:  # Form nhập tập X và tập thuộc tính
            # Lấy dữ liệu từ session
            file_path = request.session.get('file_path')
            file_name = request.session.get('file_name')  # Lấy tên file từ session
            context['data_preview'] = request.session.get('data_preview', None)  # Giữ lại preview dữ liệu

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

                except Exception as e:
                    context['error'] = f"Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
            else:
                context['error'] = "Không tìm thấy file Excel đã tải lên. Vui lòng tải lại file."
    
    return render(request, 'approximation.html', context)



def dependency(request):
    context = {}

    # Xử lý upload file Excel
    if request.method == 'POST' and 'excel_file' in request.FILES:
        excel_file = request.FILES['excel_file']

        # Lưu file Excel vào thư mục tạm
        file_path = os.path.join(settings.MEDIA_ROOT, excel_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in excel_file.chunks():
                destination.write(chunk)

        # Lưu đường dẫn file vào session
        request.session['file_path'] = file_path
        context['file_path'] = file_path
        context['file_name'] = excel_file.name

        # Xác định loại file (CSV hoặc Excel)
        try:
            file_extension = os.path.splitext(file_path)[1].lower()  # Lấy phần mở rộng
            if file_extension == '.csv':
                data = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Định dạng file không được hỗ trợ. Chỉ hỗ trợ .csv, .xls, và .xlsx.")

            # Hiển thị xem trước dữ liệu
            data_preview = data.head().to_html(classes='table table-striped')
            request.session['data_preview'] = data_preview  # Lưu vào session
            context['data_preview'] = data_preview

        except Exception as e:
            context['error'] = f"Đã xảy ra lỗi khi xử lý file: {str(e)}"

    # Xử lý nhập tập A và tập B
    elif request.method == 'POST' and 'target_setA' in request.POST and 'attributes_setB' in request.POST:
        file_path = request.session.get('file_path')

        if file_path and os.path.exists(file_path):  # Kiểm tra file đã tồn tại
            try:
                # Đọc dữ liệu từ file Excel
                data = pd.read_excel(file_path)

                # Lấy preview dữ liệu từ session
                data_preview = request.session.get('data_preview')
                if data_preview:
                    context['data_preview'] = data_preview  # Thêm preview vào context

                # Lấy tập A và tập B từ form
                target_setA = request.POST.get('target_setA')
                attributes_setB = request.POST.get('attributes_setB')

                if not target_setA or not attributes_setB:
                    raise ValueError('Vui lòng nhập đầy đủ tập A và tập B!')

                target_setA = target_setA.split(',')
                attributes_setB = attributes_setB.split(',')

                # Tạo các lớp tương đương
                equivalence_classes_A = create_equivalence_classes(data, target_setA)
                equivalence_classes_B = create_equivalence_classes(data, attributes_setB)

                # Tính xấp xỉ dưới và độ phụ thuộc thuộc tính k
                lower_approximations = {}
                total_lower_count = 0

                for class_name, class_objects in equivalence_classes_A.items():
                    lower = lower_approximation(class_objects, equivalence_classes_B)
                    lower_approximations[class_name] = lower
                    total_lower_count += len(lower)

                total_objects = sum(len(v) for v in equivalence_classes_A.values())
                dependency_k = total_lower_count / total_objects if total_objects > 0 else 0

                # Đưa kết quả vào context
                context['equivalence_classes_A'] = equivalence_classes_A
                context['equivalence_classes_B'] = equivalence_classes_B
                context['lower_approximations'] = lower_approximations
                context['dependency_k'] = dependency_k

                # Lưu lại giá trị để hiển thị khi reload
                context['target_setA'] = ','.join(target_setA)
                context['attributes_setB'] = ','.join(attributes_setB)

            except Exception as e:
                context['error'] = f"Đã xảy ra lỗi trong quá trình xử lý: {str(e)}"
        else:
            context['error'] = "Không tìm thấy file Excel đã tải lên. Vui lòng tải lại file."

    return render(request, 'dependency.html', context)




def create_equivalence_classes(data, attributes):
    """
    Tạo các lớp tương đương dựa trên danh sách thuộc tính.
    """
    equivalence_classes = {}
    for index, row in data.iterrows():
        key = tuple(row[attr] for attr in attributes)
        if key not in equivalence_classes:
            equivalence_classes[key] = []
        equivalence_classes[key].append(f'o{index + 1}')
    return equivalence_classes


def lower_approximation(class_objects, equivalence_classes_B):
    """
    Tính xấp xỉ dưới của một lớp tương đương từ tập A qua tập B.
    """
    lower = set()
    for key, objects in equivalence_classes_B.items():
        if set(objects).issubset(class_objects):
            lower.update(objects)
    return list(lower)



def reduct(request):
    context = {}

    if request.method == 'POST' and 'excel_file' in request.FILES:
        # Tải file Excel lên
        excel_file = request.FILES['excel_file']
        file_path = os.path.join(settings.MEDIA_ROOT, excel_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in excel_file.chunks():
                destination.write(chunk)

        # Lưu đường dẫn file vào session
        request.session['file_path'] = file_path
        context['file_path'] = file_path
        context['file_name'] = excel_file.name

        # Đọc dữ liệu từ file Excel
        df = pd.read_excel(file_path)

        # Hiển thị preview dữ liệu nếu đã tải file lên
        if file_path and os.path.exists(file_path):
            try:
                # Xác định định dạng file
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == '.csv':
                    data = pd.read_csv(file_path)
                elif file_extension in ['.xls', '.xlsx']:
                    data = pd.read_excel(file_path)
                else:
                    raise ValueError("Định dạng file không được hỗ trợ.")

                # Tạo bảng preview
                data_preview = data.head(10).to_html(classes='table table-striped')  # Hiển thị 10 dòng đầu
                context['data_preview'] = data_preview  # Thêm vào context

            except Exception as e:
                context['error'] = f"Không thể hiển thị dữ liệu: {str(e)}"

        # Loại bỏ cột đầu tiên để không xét nó khi tìm rút gọn
        columns = df.columns
        decision_column = columns[-1]  # Cột quyết định
        attributes = columns[1:-1]  # Các thuộc tính (bỏ cột đầu tiên và cột quyết định)

        # Hàm tìm tập rút gọn (reduct)
        def find_reducts(df, decision_column, attributes):
            reducts = []  

            # Tìm tất cả các tổ hợp thuộc tính
            for r in range(1, len(attributes) + 1):
                for subset in combinations(attributes, r):
                    grouped = df.groupby(list(subset))[decision_column].nunique()
                    if grouped.eq(1).all():
                        reducts.append(set(subset))  # Lưu tập rút gọn

            # Loại bỏ tập dư thừa (chỉ giữ tập rút gọn tối thiểu)
            minimal_reducts = []
            for reduct in reducts:
                if not any(reduct > other for other in reducts if reduct != other):
                    minimal_reducts.append(reduct)
            return minimal_reducts

        # Hàm tạo luật phân lớp
        def generate_classification(df, reduct, decision_column):
            rules = []
            for _, subset in df.groupby(list(reduct)):
                decision_values = subset[decision_column].unique()
                if len(decision_values) == 1:
                    decision_value = decision_values[0]
                    conditions = " AND ".join([f"{col} = '{subset[col].iloc[0]}'" for col in reduct])
                    rules.append(f"IF {conditions} THEN {decision_column} = '{decision_value}'")
            return rules

        # Tìm các tập rút gọn
        reducts = find_reducts(df, decision_column, attributes)

        # Đưa kết quả vào context
        context['reducts'] = [list(reduct) for reduct in reducts]

        if reducts:
            # Chọn tập rút gọn đầu tiên để tạo luật phân lớp
            chosen_reduct = list(reducts[0])
            classification_rules = generate_classification(df, chosen_reduct, decision_column)
            context['classification_rules'] = classification_rules[:3]  # Hiển thị 3 luật đầu tiên
        else:
            context['classification_rules'] = None
            context['error'] = "Không tìm thấy tập rút gọn."

    elif request.method == 'POST' and 'file_path' in request.session:
        context['error'] = "Vui lòng tải lên file Excel trước."

    return render(request, 'reduct.html', context)

def decision_tree_gain(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            steps = []  # Lưu trữ các bước xử lý
            calculations = []  # Lưu chi tiết các phép tính
            if_then_rules = []  # Lưu các luật IF-THEN

            # Bước 1: Đọc tệp đã tải lên
            file = request.FILES['file']
            file_extension = os.path.splitext(file.name)[1]
            steps.append("Tệp đã được tải lên thành công.")

            if file_extension == '.csv':
                data = pd.read_csv(file)
            elif file_extension in ['.xls', '.xlsx']:
                data = pd.read_excel(file)
            else:
                steps.append("Lỗi: Tệp phải ở định dạng CSV hoặc Excel.")
                return render(request, 'gain.html', {'error': 'Vui lòng tải lên tệp CSV hoặc Excel!', 'steps': steps})

            steps.append("Dữ liệu đã được đọc thành công từ tệp.")

            # Hiển thị xem trước dữ liệu
            data_preview = data.head().to_html(classes='table table-striped')

            # Bước 2: Chia dữ liệu thành đặc trưng (X) và mục tiêu (y)
            X = data.iloc[:, :-1]  # Tất cả các cột trừ cột cuối
            y = data.iloc[:, -1]  # Cột cuối cùng
            steps.append("Dữ liệu đã được chia thành đặc trưng (X) và mục tiêu (y).")

            # Bước 3: Mã hóa dữ liệu dạng phân loại
            label_encoders = {}
            for column in X.columns:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le

            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)
            steps.append("Dữ liệu phân loại đã được mã hóa.")

            # Hàm tính entropy
            def calculate_entropy(column):
                counts = column.value_counts(normalize=True)
                entropy = -sum(counts * np.log2(counts))
                return entropy

            # Hàm tính độ lợi thông tin
            def calculate_information_gain(data, feature, target):
                total_entropy = calculate_entropy(data[target])

                # Entropy có trọng số của các tập con
                values = data[feature].unique()
                weighted_entropy = 0
                calculations.append(f"\nTính độ lợi thông tin cho đặc trưng: {feature}")
                calculations.append(f"Entropy(S) = -Σ(p * log2(p)) với p là xác suất của các lớp trong mục tiêu")
                calculations.append(f"Entropy(S) = {total_entropy:.4f}")

                for value in values:
                    subset = data[data[feature] == value]
                    prob = len(subset) / len(data)
                    subset_entropy = calculate_entropy(subset[target])
                    weighted_entropy += prob * subset_entropy
                    calculations.append(f"\nTập con {feature}={value}: Kích thước={len(subset)}, Xác suất={prob:.4f}, Entropy={subset_entropy:.4f}")

                info_gain = total_entropy - weighted_entropy
                calculations.append(f"\nEntropy có trọng số = Σ(prob * Entropy(tập con)) = {weighted_entropy:.4f}")
                calculations.append(f"\nĐộ lợi thông tin = Entropy(S) - Entropy có trọng số = {info_gain:.4f}")
                return total_entropy, weighted_entropy, info_gain

            # Bước 4: Tính toán entropy và độ lợi thông tin cho tất cả các đặc trưng
            data['Target'] = y  # Thêm cột mục tiêu vào DataFrame để tính toán
            best_feature = None
            best_info_gain = -1

            for feature in X.columns:
                total_entropy, weighted_entropy, info_gain = calculate_information_gain(data, feature, 'Target')
                calculations.append(
                    f"Đặc trưng: {feature} -> Entropy(S) = {total_entropy:.4f}, Entropy có trọng số = {weighted_entropy:.4f}, Độ lợi thông tin = {info_gain:.4f}"
                )

                if info_gain > best_info_gain:
                    best_feature = feature
                    best_info_gain = info_gain

            steps.append(f"Đặc trưng được chọn với độ lợi thông tin cao nhất: {best_feature} (Gain = {best_info_gain:.4f}).")

            # Bước 5: Huấn luyện cây quyết định bằng ID3
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
            clf_entropy.fit(X_train, y_train)
            steps.append("Mô hình cây quyết định đã được huấn luyện bằng tiêu chí entropy.")

            # Bước 6: Dự đoán và tính độ chính xác
            y_pred = clf_entropy.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            steps.append(f"Dự đoán đã được thực hiện. Độ chính xác của mô hình: {accuracy:.2f}.")

            # Bước 7: Xuất cấu trúc cây và tạo các luật IF-THEN
            tree_rules = export_text(clf_entropy, feature_names=list(X.columns))
            steps.append("Cấu trúc cây quyết định đã được xuất.")

            # Tạo các luật IF-THEN
            def generate_rules(tree, feature_names, target_names):
                tree_ = tree.tree_
                feature_name = [
                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                ]

                paths = []

                def recurse(node, path, paths):
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                        name = feature_name[node]
                        threshold = tree_.threshold[node]
                        left_path = path + [f"({name} <= {threshold})"]
                        right_path = path + [f"({name} > {threshold})"]
                        recurse(tree_.children_left[node], left_path, paths)
                        recurse(tree_.children_right[node], right_path, paths)
                    else:
                        path_str = " AND ".join(path)
                        class_name = target_names[np.argmax(tree_.value[node])]
                        paths.append(f"IF {path_str} THEN {class_name}")

                recurse(0, [], paths)
                return paths

            if_then_rules = generate_rules(clf_entropy, list(X.columns), y_encoder.classes_)

            # Bước 8: Vẽ và lưu hình ảnh cây quyết định
            plt.figure(figsize=(20, 10))
            plot_tree(
                clf_entropy,
                feature_names=X.columns,
                class_names=y_encoder.classes_,
                filled=True
            )
            image_path = 'static/decision_tree_entropy.png'
            plt.savefig(image_path)
            plt.close()
            steps.append("Hình ảnh cây quyết định đã được lưu.")

            # Truyền dữ liệu sang giao diện
            context = {
                'steps': steps,
                'calculations': calculations,
                'tree_rules': tree_rules,
                'if_then_rules': if_then_rules,
                'accuracy': accuracy,
                'results': f'Độ chính xác của mô hình: {accuracy:.2f}',
                'image_path': f'/{image_path}',
                'data_preview': data_preview,
            }
            return render(request, 'gain.html', context)

        except Exception as e:
            steps.append(f"Lỗi xử lý: {str(e)}")
            return render(request, 'gain.html', {'error': f'Lỗi xử lý tệp: {str(e)}', 'steps': steps})

    return render(request, 'gain.html')




def decision_tree_gini(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            steps = []  # Store steps
            calculations = []  # Store detailed calculations
            if_then_rules = []  # Store IF-THEN rules

            # Step 1: Read uploaded file
            file = request.FILES['file']
            file_extension = os.path.splitext(file.name)[1]
            steps.append("File uploaded successfully.")

            if file_extension == '.csv':
                data = pd.read_csv(file)
            elif file_extension in ['.xls', '.xlsx']:
                data = pd.read_excel(file)
            else:
                steps.append("Error: File must be in CSV or Excel format.")
                return render(request, 'gini.html', {'error': 'Please upload a CSV or Excel file!', 'steps': steps})

            steps.append("Data successfully loaded from the file.")

            # Preview data
            data_preview = data.head().to_html(classes='table table-striped')

            # Step 2: Split data into features (X) and target (y)
            X = data.iloc[:, :-1]  # All columns except the last
            y = data.iloc[:, -1]  # The last column
            steps.append("Data split into features (X) and target (y).")

            # Step 3: Encode categorical data
            label_encoders = {}
            for column in X.columns:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le

            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)
            steps.append("Categorical data encoded successfully.")

            # Function to calculate Gini Index
            def calculate_gini_index(column):
                counts = column.value_counts(normalize=True)
                gini = 1 - sum(counts ** 2)
                return gini

            # Function to calculate Gini Gain
            def calculate_gini_gain(data, feature, target):
                total_gini = calculate_gini_index(data[target])

                # Weighted Gini of subsets
                values = data[feature].unique()
                weighted_gini = 0
                calculations.append(f"\nCalculating Gini Gain for feature: {feature}")
                calculations.append(f"Gini(S) = 1 - Σ(p²), where p is the probability of each class in the target")
                calculations.append(f"Gini(S) = {total_gini:.4f}")

                for value in values:
                    subset = data[data[feature] == value]
                    prob = len(subset) / len(data)
                    subset_gini = calculate_gini_index(subset[target])
                    weighted_gini += prob * subset_gini
                    calculations.append(f"\nSubset {feature}={value}: Size={len(subset)}, Probability={prob:.4f}, Gini={subset_gini:.4f}")

                gini_gain = total_gini - weighted_gini
                calculations.append(f"\nWeighted Gini = Σ(prob * Gini(subset)) = {weighted_gini:.4f}")
                calculations.append(f"\nGini Gain = Gini(S) - Weighted Gini = {gini_gain:.4f}")
                return total_gini, weighted_gini, gini_gain

            # Step 4: Calculate Gini and Gini Gain for all features
            data['Target'] = y  # Add target column to DataFrame for calculations
            best_feature = None
            best_gini_gain = -1

            for feature in X.columns:
                total_gini, weighted_gini, gini_gain = calculate_gini_gain(data, feature, 'Target')
                calculations.append(
                    f"Feature: {feature} -> Gini(S) = {total_gini:.4f}, Weighted Gini = {weighted_gini:.4f}, Gini Gain = {gini_gain:.4f}"
                )

                if gini_gain > best_gini_gain:
                    best_feature = feature
                    best_gini_gain = gini_gain

            steps.append(f"Best feature selected with highest Gini Gain: {best_feature} (Gain = {best_gini_gain:.4f}).")

            # Step 5: Train decision tree using Gini criterion
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
            clf_gini.fit(X_train, y_train)
            steps.append("Decision tree model trained using Gini Index criterion.")

            # Step 6: Predict and calculate accuracy
            y_pred = clf_gini.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            steps.append(f"Predictions made. Model accuracy: {accuracy:.2f}.")

            # Step 7: Export tree structure and create IF-THEN rules
            tree_rules = export_text(clf_gini, feature_names=list(X.columns))
            steps.append("Decision tree structure exported.")

            # Generate IF-THEN rules
            def generate_rules(tree, feature_names, target_names):
                tree_ = tree.tree_
                feature_name = [
                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                ]

                paths = []

                def recurse(node, path, paths):
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                        name = feature_name[node]
                        threshold = tree_.threshold[node]
                        left_path = path + [f"({name} <= {threshold})"]
                        right_path = path + [f"({name} > {threshold})"]
                        recurse(tree_.children_left[node], left_path, paths)
                        recurse(tree_.children_right[node], right_path, paths)
                    else:
                        path_str = " AND ".join(path)
                        class_name = target_names[np.argmax(tree_.value[node])]
                        paths.append(f"IF {path_str} THEN {class_name}")

                recurse(0, [], paths)
                return paths

            if_then_rules = generate_rules(clf_gini, list(X.columns), y_encoder.classes_)

            # Step 8: Visualize and save decision tree image
            plt.figure(figsize=(20, 10))
            plot_tree(
                clf_gini,
                feature_names=X.columns,
                class_names=y_encoder.classes_,
                filled=True
            )
            image_path = 'static/decision_tree_gini.png'
            plt.savefig(image_path)
            plt.close()
            steps.append("Decision tree image saved.")

            # Pass data to template
            context = {
                'steps': steps,
                'calculations': calculations,
                'tree_rules': tree_rules,
                'if_then_rules': if_then_rules,
                'accuracy': accuracy,
                'results': f'Model accuracy: {accuracy:.2f}',
                'image_path': f'/{image_path}',
                'data_preview': data_preview,
            }
            return render(request, 'gini.html', context)

        except Exception as e:
            steps.append(f"Processing error: {str(e)}")
            return render(request, 'gini.html', {'error': f'File processing error: {str(e)}', 'steps': steps})

    return render(request, 'gini.html')



def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Thuật toán K-means
def kmeans(points, centroids, k, max_iters=100):
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[len(new_centroids)])

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, clusters

# Thuật toán K-means
def kmeans(points, centroids, k, max_iters=100):
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroid = np.mean(cluster, axis=0)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[len(new_centroids)])

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, clusters

def kmeans_view(request):
    context = {}

    if request.method == 'POST':
        if 'file' in request.FILES:
            try:
                file = request.FILES['file']
                df = pd.read_excel(file)
                df = df.dropna()  # Remove null values
                request.session['df'] = df.to_dict()
                context['columns'] = df.columns.tolist()
                context['file_uploaded'] = True
            except Exception as e:
                context['error'] = f"Lỗi khi đọc tệp: {str(e)}"

        elif request.POST.getlist('attributes'):
            try:
                df = pd.DataFrame(request.session['df'])
                selected_attributes = request.POST.getlist('attributes')
                if not selected_attributes:
                    context['error'] = "Vui lòng chọn các thuộc tính để phân cụm."
                    context['columns'] = df.columns.tolist()
                    return render(request, 'kmeans.html', context)

                data = df[selected_attributes].copy()
                for col in selected_attributes:
                    if data[col].dtype == 'object':
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))

                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)

                distortions = []
                K = range(1, 11)
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(data_scaled)
                    distortions.append(kmeans.inertia_)

                plt.figure(figsize=(8, 6))
                plt.plot(K, distortions, 'bx-')
                plt.xlabel('Số cụm (k)')
                plt.ylabel('Độ méo (Inertia)')
                plt.title('Phương pháp Elbow')
                elbow_buffer = io.BytesIO()
                plt.savefig(elbow_buffer, format='png')
                elbow_buffer.seek(0)
                elbow_uri = base64.b64encode(elbow_buffer.read()).decode('utf-8')
                plt.close()

                knee_locator = knee_locator(K, distortions, curve='convex', direction='decreasing')
                optimal_k = knee_locator.knee

                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(data_scaled)

                df['Cluster'] = clusters

                plt.figure(figsize=(8, 6))
                plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis')
                plt.title('Phân Cụm KMeans')
                plt.xlabel(selected_attributes[0])
                plt.ylabel(selected_attributes[1])
                cluster_buffer = io.BytesIO()
                plt.savefig(cluster_buffer, format='png')
                cluster_buffer.seek(0)
                cluster_uri = base64.b64encode(cluster_buffer.read()).decode('utf-8')
                plt.close()

                cluster_details = {}
                for cluster in np.unique(clusters):
                    cluster_data = df[df['Cluster'] == cluster][selected_attributes].to_dict('records')
                    cluster_details[f'Cụm {cluster + 1}'] = cluster_data

                silhouette_avg = metrics.silhouette_score(data_scaled, clusters)

                context.update({
                    'success': True,
                    'optimal_k': optimal_k,
                    'silhouette_score': round(silhouette_avg, 4),
                    'elbow_uri': elbow_uri,
                    'cluster_uri': cluster_uri,
                    'cluster_details': cluster_details,
                })
            except Exception as e:
                context['error'] = f"Lỗi khi chạy KMeans: {str(e)}"

    return render(request, 'kmeans.html', context)

def bayes_view(request):
    context = {}

    if request.method == 'POST':
        if 'file' in request.FILES:
            try:
                file = request.FILES['file']
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                else:
                    context['error'] = "Chỉ hỗ trợ file CSV hoặc Excel."
                    return render(request, 'bayes.html', context)

                # Loại bỏ giá trị thiếu
                df = df.dropna()

                # Lưu dữ liệu vào session
                request.session['data'] = df.to_dict()
                # Chuẩn bị danh sách cột và giá trị duy nhất cho từng cột
                unique_values = [(col, df[col].dropna().unique().tolist()) for col in df.columns]
                context['columns'] = df.columns.tolist()
                context['unique_values'] = unique_values
                context['data_uploaded'] = True

            except Exception as e:
                context['error'] = f"Lỗi khi xử lý file: {e}"

        elif 'calculate' in request.POST:
            try:
                df = pd.DataFrame(request.session['data'])

                target_column = request.POST.get('target_column')
                feature_columns = request.POST.getlist('feature_columns')
                selected_values = {col: request.POST.get(col) for col in feature_columns}

                if not target_column or not feature_columns:
                    context['error'] = "Vui lòng chọn cột quyết định và các cột tham gia."
                    unique_values = [(col, df[col].dropna().unique().tolist()) for col in df.columns]
                    context['columns'] = df.columns.tolist()
                    context['unique_values'] = unique_values
                    return render(request, 'bayes.html', context)

                total_samples = len(df)
                yes_count = len(df[df[target_column] == 'Yes'])
                no_count = len(df[df[target_column] == 'No'])

                prob_yes = yes_count / total_samples
                prob_no = no_count / total_samples

                prob_values_given_yes = {}
                prob_values_given_no = {}

                for col in feature_columns:
                    value = selected_values[col]

                    yes_value_count = len(df[(df[col] == value) & (df[target_column] == 'Yes')])
                    no_value_count = len(df[(df[col] == value) & (df[target_column] == 'No')])

                    prob_value_given_yes = yes_value_count / yes_count if yes_count > 0 else 0
                    prob_value_given_no = no_value_count / no_count if no_count > 0 else 0

                    prob_values_given_yes[col] = round(prob_value_given_yes, 4)
                    prob_values_given_no[col] = round(prob_value_given_no, 4)

                yes_product = prob_yes
                no_product = prob_no

                for col in feature_columns:
                    yes_product *= prob_values_given_yes[col]
                    no_product *= prob_values_given_no[col]

                result = "Yes" if yes_product > no_product else "No"

                context['success'] = True
                context['result'] = result
                context['prob_yes'] = round(yes_product, 4)
                context['prob_no'] = round(no_product, 4)
                context['prob_values_given_yes'] = prob_values_given_yes
                context['prob_values_given_no'] = prob_values_given_no
                context['initial_prob_yes'] = round(prob_yes, 4)
                context['initial_prob_no'] = round(prob_no, 4)

            except Exception as e:
                context['error'] = f"Lỗi khi tính toán: {e}"

    return render(request, 'bayes.html', context)

def laplace_view(request):
    context = {}

    if request.method == 'POST':
        if 'file' in request.FILES:
            try:
                file = request.FILES['file']
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.name.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                else:
                    context['error'] = "Chỉ hỗ trợ file CSV hoặc Excel."
                    return render(request, 'laplace.html', context)

                # Loại bỏ giá trị thiếu
                df = df.dropna()

                # Lưu dữ liệu vào session
                request.session['data'] = df.to_dict()
                # Chuẩn bị danh sách cột và giá trị duy nhất cho từng cột
                unique_values = [(col, df[col].dropna().unique().tolist()) for col in df.columns]
                context['columns'] = df.columns.tolist()
                context['unique_values'] = unique_values
                context['data_uploaded'] = True

            except Exception as e:
                context['error'] = f"Lỗi khi xử lý file: {e}"

        elif 'calculate' in request.POST:
            try:
                df = pd.DataFrame(request.session['data'])

                target_column = request.POST.get('target_column')
                feature_columns = request.POST.getlist('feature_columns')
                selected_values = {col: request.POST.get(col) for col in feature_columns}

                if not target_column or not feature_columns:
                    context['error'] = "Vui lòng chọn cột quyết định và các cột tham gia."
                    unique_values = [(col, df[col].dropna().unique().tolist()) for col in df.columns]
                    context['columns'] = df.columns.tolist()
                    context['unique_values'] = unique_values
                    return render(request, 'laplace.html', context)

                total_samples = len(df)
                class_counts = df[target_column].value_counts()
                classes = class_counts.index.tolist()

                # Tính xác suất ban đầu P(C) với làm trơn Laplace
                prob_classes = {
                    cls: (class_counts[cls] + 1) / (total_samples + len(classes))
                    for cls in classes
                }

                prob_values_given_class = {cls: {} for cls in classes}
                class_products = {cls: prob_classes[cls] for cls in classes}

                for col in feature_columns:
                    value = selected_values[col]
                    unique_values = df[col].nunique()

                    for cls in classes:
                        value_count = len(df[(df[col] == value) & (df[target_column] == cls)])

                        # Tính xác suất có làm trơn Laplace
                        prob_value_given_class = (value_count + 1) / (class_counts[cls] + unique_values)
                        prob_values_given_class[cls][col] = round(prob_value_given_class, 4)

                        # Nhân xác suất
                        class_products[cls] *= prob_value_given_class

                # Kết quả dự đoán
                result = max(class_products, key=class_products.get)

                # Gửi dữ liệu ra giao diện
                context['success'] = True
                context['result'] = result
                context['prob_classes'] = {cls: round(prob, 4) for cls, prob in prob_classes.items()}
                context['class_products'] = {cls: round(product, 4) for cls, product in class_products.items()}
                context['prob_values_given_class'] = prob_values_given_class

            except Exception as e:
                context['error'] = f"Lỗi khi tính toán: {e}"

    return render(request, 'laplace.html', context)

def kohonen_view(request):
    context = {}

    if request.method == 'POST':
        if request.FILES.get('file'):
            try:
                # Đọc tệp Excel
                file = request.FILES['file']
                df = pd.read_excel(file)

                # Lưu dữ liệu vào session
                request.session['df'] = df.to_dict()
                context['columns'] = df.columns.tolist()
                context['file_uploaded'] = True

            except Exception as e:
                context['error'] = f"Lỗi khi đọc tệp: {str(e)}"

        elif request.POST.getlist('attributes'):
            try:
                # Lấy dữ liệu từ session
                df = pd.DataFrame(request.session['df'])

                # Lấy các thuộc tính được chọn
                selected_attributes = request.POST.getlist('attributes')
                if not selected_attributes:
                    context['error'] = "Vui lòng chọn các thuộc tính để phân cụm."
                    context['columns'] = df.columns.tolist()
                    return render(request, 'kohonen.html', context)

                # Tiền xử lý dữ liệu
                data = df[selected_attributes].copy()
                for col in selected_attributes:
                    if data[col].dtype == 'object':
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))

                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)

                # Khởi tạo lưới Kohonen
                grid_shape = (5, 5)
                learning_rate = 0.5
                radius = 1.0
                epochs = 100
                grid = np.random.rand(grid_shape[0], grid_shape[1], data_scaled.shape[1])

                # Huấn luyện Kohonen
                for epoch in range(epochs):
                    for sample in data_scaled:
                        # Tìm neuron chiến thắng
                        distances = np.linalg.norm(grid - sample, axis=2)
                        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)

                        # Cập nhật trọng số
                        for i in range(grid.shape[0]):
                            for j in range(grid.shape[1]):
                                dist_to_bmu = np.sqrt((i - bmu_idx[0]) ** 2 + (j - bmu_idx[1]) ** 2)
                                if dist_to_bmu <= radius:
                                    influence = np.exp(-dist_to_bmu / (2 * (radius ** 2)))
                                    grid[i, j] += learning_rate * influence * (sample - grid[i, j])

                # Dự đoán cụm
                predictions = []
                for sample in data_scaled:
                    distances = np.linalg.norm(grid - sample, axis=2)
                    bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    predictions.append(bmu_idx)

                # Thêm kết quả phân cụm vào DataFrame
                df['Cluster'] = [f"({p[0]}, {p[1]})" for p in predictions]

                # Trực quan hóa kết quả phân cụm
                plt.figure(figsize=(8, 6))
                plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=[p[0] * grid_shape[1] + p[1] for p in predictions], cmap='viridis')
                plt.title('Phân Cụm Kohonen')
                plt.xlabel(selected_attributes[0])
                plt.ylabel(selected_attributes[1])
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                img_uri = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()

                # Chi tiết cụm
                cluster_details = {}
                for cluster in df['Cluster'].unique():
                    cluster_details[cluster] = df[df['Cluster'] == cluster][selected_attributes].to_dict('records')

                # Trả kết quả ra giao diện
                context.update({
                    'success': True,
                    'img_uri': img_uri,
                    'cluster_details': cluster_details
                })

            except Exception as e:
                context['error'] = f"Lỗi khi chạy Kohonen: {str(e)}"

    return render(request, 'kohonen.html', context)