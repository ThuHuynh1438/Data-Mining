import base64
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
import os
from math import log2
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
                    'equation': f"y = {b1:.3f}x + {b0:.3f}",
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

    # Xử lý nhập tập A và tập B
    elif request.method == 'POST' and 'target_setA' in request.POST and 'attributes_setB' in request.POST:
        file_path = request.session.get('file_path')

        if file_path and os.path.exists(file_path):  # Kiểm tra file đã tồn tại
            try:
                # Đọc dữ liệu từ file Excel
                data = pd.read_excel(file_path)

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

# Hàm tính Entropy của dữ liệu
def calculate_entropy(data, target_column):
    # Tính tần suất của các giá trị trong cột mục tiêu
    value_counts = data[target_column].value_counts(normalize=True)
    
    # Tính Entropy
    entropy = -sum(value_counts * np.log2(value_counts))
    return entropy

# Hàm tính Information Gain
def calculate_information_gain(data, feature_column, target_column):
    total_entropy = calculate_entropy(data, target_column)
    feature_values = data[feature_column].value_counts(normalize=True)
    weighted_entropy = 0
    for feature_value, weight in feature_values.items():
        subset = data[data[feature_column] == feature_value]
        weighted_entropy += weight * calculate_entropy(subset, target_column)
    info_gain = total_entropy - weighted_entropy
    return info_gain


def decision_tree_gain(request):
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
                return render(request, 'gain.html', {'error': 'Please upload a CSV or Excel file!', 'steps': steps})

            steps.append("Data successfully loaded from the file.")

            # Preview data
            data_preview = data.to_html(classes='table table-striped')

            # Step 2: Calculate Information Gain for each feature
            target_column = data.columns[-1]  # Assuming the last column is the target
            info_gain_results = []
            for feature in data.columns[:-1]:  # Exclude the target column
                info_gain = calculate_information_gain(data, feature, target_column)
                info_gain_results.append(f"Information Gain for '{feature}': {info_gain:.4f}")

            calculations = "\n".join(info_gain_results)

            # Step 3: Train decision tree and generate rules & image
            def decision_tree_algorithm(data):
                # Encode categorical features
                encoders = {}
                for col in data.columns:
                    if data[col].dtype == 'object':
                        encoder = LabelEncoder()
                        data[col] = encoder.fit_transform(data[col])
                        encoders[col] = encoder

                X = data.iloc[:, :-1]  # Features
                y = data.iloc[:, -1]   # Target

                # Train Decision Tree model
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X, y)

                # Generate IF-THEN rules
                if_then_rules = generate_if_then_rules(model, list(X.columns))

                # Plot and save the tree image for Gain
                plt.figure(figsize=(12, 8))
                plot_tree(model, feature_names=list(X.columns), class_names=model.classes_.astype(str), filled=True, rounded=True, fontsize=10)
                plt.title("Decision Tree Visualization for Gain")

                # Save the tree image for Gain
                image_filename = 'decision_tree_gain.png'  # Save image with 'gain' in the filename
                image_path = f"images/{image_filename}"  # Relative path from static directory
                os.makedirs(os.path.join(settings.BASE_DIR, 'static', 'images'), exist_ok=True)
                plt.savefig(os.path.join(settings.BASE_DIR, 'static', image_path))  # Save the image to the static folder
                plt.close()

                return if_then_rules, image_path

            if_then_rules, image_path = decision_tree_algorithm(data)
            steps.append("Decision tree model trained and image saved.")

            # Pass data to template
            context = {
                'steps': steps,
                'calculations': calculations,
                'if_then_rules': if_then_rules,  # Pass the IF-THEN rules
                'tree_rules': '\n'.join(if_then_rules),  # Show IF-THEN rules
                'image_path': image_path,  # Pass the relative path to the image
                'data_preview': data_preview,
            }
            return render(request, 'gain.html', context)

        except Exception as e:
            steps.append(f"Processing error: {str(e)}")
            return render(request, 'gain.html', {'error': f'File processing error: {str(e)}', 'steps': steps})

    return render(request, 'gain.html')


def generate_if_then_rules(model, feature_names):
    tree_rules = []
    # Truy cập vào cấu trúc của cây quyết định
    tree_ = model.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    def recurse(node, rule):
        # Nếu đây là một lá cây (terminal node), tức là có dự đoán kết quả
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            tree_rules.append(f"{rule} THEN class = {tree_.value[node].argmax()}")
        else:
            # Nếu node này là một quyết định, kiểm tra điều kiện
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left = tree_.children_left[node]
            right = tree_.children_right[node]
            
            # Tạo quy tắc IF-THEN cho node hiện tại
            recurse(left, f"{rule} IF {name} <= {threshold}")
            recurse(right, f"{rule} IF {name} > {threshold}")

    recurse(0, "")  # Bắt đầu từ node gốc
    return tree_rules


# Hàm tính Gini Impurity của dữ liệu
def calculate_gini(data, target_column):
    value_counts = data[target_column].value_counts(normalize=True)
    gini = 1 - sum(value_counts**2)
    return gini

# Hàm tính Gini Impurity cho các thuộc tính trong cây quyết định
def calculate_feature_gini(model, X, feature_names):
    feature_ginis = {}
    tree_ = model.tree_
    
    # Lấy Gini của mỗi thuộc tính từ các node của cây quyết định
    for i, feature_name in enumerate(feature_names):
        feature_index = X.columns.get_loc(feature_name)
        gini_values = tree_.impurity[tree_.feature == feature_index]
        if len(gini_values) > 0:
            feature_ginis[feature_name] = np.mean(gini_values)  # Tính trung bình Gini của các node có thuộc tính này
        else:
            feature_ginis[feature_name] = 0.0  # Nếu không có phân chia cho thuộc tính này

    return feature_ginis

# Hàm trích xuất IF-THEN Rules từ cây quyết định
def generate_if_then_rules(model, feature_names):
    tree_rules = []
    tree_ = model.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    def recurse(node, rule):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            tree_rules.append(f"{rule} THEN class = {tree_.value[node].argmax()}")
        else:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left = tree_.children_left[node]
            right = tree_.children_right[node]
            recurse(left, f"{rule} IF {name} <= {threshold}")
            recurse(right, f"{rule} IF {name} > {threshold}")

    recurse(0, "")  # Bắt đầu từ node gốc
    return tree_rules

def decision_tree_gini(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            steps = []  # Store steps
            calculations = []  # Store detailed calculations
            if_then_rules = []  # Store IF-THEN rules
            feature_ginis = {}  # Store Gini Impurity of features

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
            data_preview = data.to_html(classes='table table-striped')

            # Step 2: Train decision tree and generate rules & image
            def decision_tree_algorithm(data):
                # Encode categorical features
                encoders = {}
                for col in data.columns:
                    if data[col].dtype == 'object':
                        encoder = LabelEncoder()
                        data[col] = encoder.fit_transform(data[col])
                        encoders[col] = encoder

                X = data.iloc[:, :-1]  # Features
                y = data.iloc[:, -1]   # Target

                # Train Decision Tree model
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X, y)

                # Generate IF-THEN rules
                if_then_rules = generate_if_then_rules(model, list(X.columns))

                # Calculate Gini Impurity for each feature
                feature_ginis = calculate_feature_gini(model, X, list(X.columns))

                # Plot and save the tree image
                plt.figure(figsize=(12, 8))
                plot_tree(model, feature_names=list(X.columns), class_names=model.classes_.astype(str), filled=True, rounded=True, fontsize=10)
                plt.title("Decision Tree Visualization")

                # Save the tree image
                image_filename = 'decision_tree_plot.png'
                image_path = f"images/{image_filename}"  # Relative path from static directory
                os.makedirs(os.path.join(settings.BASE_DIR, 'static', 'images'), exist_ok=True)
                plt.savefig(os.path.join(settings.BASE_DIR, 'static', image_path))  # Save the image to the static folder
                plt.close()

                return if_then_rules, image_path, feature_ginis

            if_then_rules, image_path, feature_ginis = decision_tree_algorithm(data)
            steps.append("Decision tree model trained and image saved.")

            # Pass data to template
            context = {
                'steps': steps,
                'calculations': calculations,
                'if_then_rules': if_then_rules,  # Pass the IF-THEN rules
                'feature_ginis': feature_ginis,  # Pass Gini Impurity of features
                'image_path': image_path,  # Pass the relative path to the image
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