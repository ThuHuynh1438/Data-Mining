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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
from io import BytesIO
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
# Tính Gini
def calculate_gini(column):
    values, counts = np.unique(column, return_counts=True)
    total = sum(counts)
    return 1 - sum((count / total) ** 2 for count in counts)

def calculate_gini_for_attribute(data, attribute, target):
    total = len(data)
    gini_index = 0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        gini_index += (len(subset) / total) * calculate_gini(subset[target])
    return gini_index

def build_tree_gini(data, target, attributes):
    tree = {}
    unique_classes = np.unique(data[target])
    
    # Nếu chỉ có một lớp kết quả, trả về lớp đó
    if len(unique_classes) == 1:
        return unique_classes[0]
    
    # Nếu không còn thuộc tính nào để chia, trả về lớp có số lượng lớn nhất
    if not attributes:
        return data[target].mode()[0]

    # Tính Gini cho từng thuộc tính
    gini_values = [calculate_gini_for_attribute(data, attr, target) for attr in attributes]
    best_attr = attributes[np.argmin(gini_values)]
    tree[best_attr] = {}

    # Chia dữ liệu theo giá trị của thuộc tính tốt nhất và tiếp tục xây dựng cây
    for value, subset in data.groupby(best_attr):
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = build_tree_gini(subset, target, remaining_attrs)

    return tree

# Hàm vẽ cây quyết định sử dụng Gini
def draw_tree_gini(tree, ax, x=0, y=0, dx=1, dy=1):
    if isinstance(tree, dict):
        root = list(tree.keys())[0]
        children = tree[root]
        ax.text(x, y, root, ha='center', va='center', bbox=dict(facecolor='skyblue', edgecolor='black'))
        n = len(children)
        for i, (edge, subtree) in enumerate(children.items()):
            child_x = x - dx * (n - 1) / 2 + i * dx
            child_y = y - dy
            ax.plot([x, child_x], [y, child_y], 'k-')
            ax.text((x + child_x) / 2, (y + child_y) / 2, edge, color='red')
            draw_tree_gini(subtree, ax, child_x, child_y, dx / 2, dy)
    else:
        ax.text(x, y, tree, ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))

# Trích xuất các luật từ cây quyết định sử dụng Gini
def extract_rules_gini(tree, current_rule="", rules=None):
    if rules is None:
        rules = []
    if isinstance(tree, dict):
        root = list(tree.keys())[0]
        for value, subtree in tree[root].items():
            new_rule = f"{current_rule} ({root}={value})"
            extract_rules_gini(subtree, new_rule, rules)
    else:
        rules.append(f"{current_rule} THEN {tree}")
    return rules

# Hàm xử lý request và xây dựng cây quyết định bằng Gini
def gini(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            # Đọc file Excel và xây dựng cây
            data = pd.read_excel(full_path)
            target = data.columns[-1]  # Cột mục tiêu là cột cuối cùng
            attributes = list(data.columns[:-1])  # Các thuộc tính là các cột trước cột mục tiêu

            # Tính chỉ số Gini cho từng thuộc tính
            gini_values = {
                attr: calculate_gini_for_attribute(data, attr, target)
                for attr in attributes
            }

            # Tạo cây quyết định và vẽ cây
            tree = build_tree_gini(data, target, attributes)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            draw_tree_gini(tree, ax)

            # Lưu hình ảnh cây quyết định
            image_path = os.path.join(settings.MEDIA_ROOT, 'decision_tree_gini.png')
            plt.savefig(image_path, bbox_inches='tight')
            image_path = fs.url('decision_tree_gini.png')

            # Trích xuất các quy tắc
            rules = extract_rules_gini(tree)

            return render(request, 'gini.html', {
                'image_url': image_path,
                'rules': "\n".join(rules),
                'gini_values': gini_values
            })

        except Exception as e:
            return render(request, 'gini.html', {'error': f'Error processing file: {str(e)}'})

    return render(request, 'gini.html')

# Tính entropy
def entropy(data):
    total = len(data)
    value_counts = data.value_counts()
    return -sum((count / total) * log2(count / total) for count in value_counts if count > 0)

# Tính Information Gain cho thuộc tính
def information_gain(data, attribute, target):
    # Entropy của dữ liệu trước khi chia
    entropy_before = entropy(data[target])

    # Entropy của dữ liệu sau khi chia theo thuộc tính
    subsets = data.groupby(attribute)
    total = len(data)
    weighted_entropy = sum((len(subset) / total) * entropy(subset[target]) for _, subset in subsets)

    # Gain = Entropy ban đầu - Entropy sau khi chia
    return entropy_before - weighted_entropy

# Xây dựng cây quyết định sử dụng Gain
def build_tree_gain(data, target, attributes):
    tree = {}
    unique_classes = data[target].unique()
    
    # Nếu chỉ có một lớp kết quả, trả về lớp đó
    if len(unique_classes) == 1:
        return unique_classes[0]
    
    # Nếu không còn thuộc tính nào để chia, trả về lớp có số lượng lớn nhất
    if not attributes:
        return data[target].mode()[0]

    # Tính Information Gain cho từng thuộc tính
    gains = [information_gain(data, attr, target) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]
    tree[best_attr] = {}

    # Chia dữ liệu theo giá trị của thuộc tính tốt nhất và tiếp tục xây dựng cây
    for value, subset in data.groupby(best_attr):
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = build_tree_gain(subset, target, remaining_attrs)

    return tree

# Hàm vẽ cây quyết định
def draw_tree_gain(tree, ax, x=0, y=0, dx=1, dy=1):
    if isinstance(tree, dict):
        root = list(tree.keys())[0]
        children = tree[root]
        ax.text(x, y, root, ha='center', va='center', bbox=dict(facecolor='skyblue', edgecolor='black'))
        n = len(children)
        for i, (edge, subtree) in enumerate(children.items()):
            child_x = x - dx * (n - 1) / 2 + i * dx
            child_y = y - dy
            ax.plot([x, child_x], [y, child_y], 'k-')
            ax.text((x + child_x) / 2, (y + child_y) / 2, edge, color='red')
            draw_tree_gain(subtree, ax, child_x, child_y, dx / 2, dy)
    else:
        ax.text(x, y, tree, ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))

# Trích xuất các luật từ cây quyết định
def extract_rules_gain(tree, current_rule="", rules=None):
    if rules is None:
        rules = []
    if isinstance(tree, dict):
        root = list(tree.keys())[0]
        for value, subtree in tree[root].items():
            new_rule = f"{current_rule} ({root}={value})"
            extract_rules_gain(subtree, new_rule, rules)
    else:
        rules.append(f"{current_rule} THEN {tree}")
    return rules

# Hàm xử lý request và xây dựng cây quyết định bằng Gain
def gain(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            # Đọc file Excel và xây dựng cây
            data = pd.read_excel(full_path)
            target = data.columns[-1]  # Cột mục tiêu là cột cuối cùng
            attributes = list(data.columns[:-1])  # Các thuộc tính là các cột trước cột mục tiêu

            # Tính chỉ số Gain cho từng thuộc tính
            gain_values = {
                attr: information_gain(data, attr, target)
                for attr in attributes
            }

            # Tạo cây quyết định và vẽ cây
            tree = build_tree_gain(data, target, attributes)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('off')
            draw_tree_gain(tree, ax)

            # Lưu hình ảnh cây quyết định
            image_path = os.path.join(settings.MEDIA_ROOT, 'decision_tree_gain.png')
            plt.savefig(image_path, bbox_inches='tight')
            image_path = fs.url('decision_tree_gain.png')

            # Trích xuất các quy tắc
            rules = extract_rules_gain(tree)

            return render(request, 'gain.html', {
                'image_url': image_path,
                'rules': "\n".join(rules),
                'gain_values': gain_values
            })

        except Exception as e:
            return render(request, 'gain.html', {'error': f'Error processing file: {str(e)}'})

    return render(request, 'gain.html')

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def read_file(file):
    """Hàm đọc file dữ liệu: CSV, Excel, TXT."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file).values
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            return pd.read_excel(file).values
        elif file.name.endswith('.txt'):
            return pd.read_csv(file, delimiter="\t").values
        else:
            raise ValueError("File phải có định dạng .csv, .xlsx hoặc .txt")
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file: {e}")

def kmeans(points, k=2, epsilon=1e-6, max_iter=100):
    """
    Thuật toán K-Means với chi tiết từng lần lặp.
    """
    n = len(points)
    U = np.zeros((k, n))
    U[0, 0] = 1  # Gán điểm đầu tiên vào cụm 1
    U[1, 1:] = 1  # Gán các điểm còn lại vào cụm 2

    points = np.array(points)
    centroids = np.zeros((k, points.shape[1]))  # Trọng tâm các cụm
    iterations = []  # Danh sách lưu thông tin từng vòng lặp

    for iteration in range(max_iter):
        # Bước 1: Tính trọng tâm cụm
        for i in range(k):
            indices = np.where(U[i] == 1)[0]
            if len(indices) > 0:
                centroids[i] = np.mean(points[indices], axis=0)

        # Bước 2: Tính khoảng cách và gán điểm vào cụm gần nhất
        new_U = np.zeros_like(U)
        for j, point in enumerate(points):
            distances = [np.linalg.norm(point - centroids[i]) for i in range(k)]
            closest_cluster = np.argmin(distances)
            new_U[closest_cluster, j] = 1

        # Tính sự thay đổi giữa ma trận mới và cũ
        diff = np.linalg.norm(new_U - U)

        # Lưu thông tin vòng lặp
        iterations.append({
            'iteration': iteration + 1,
            'centroids': centroids.copy(),
            'U': new_U.copy(),
            'diff': diff
        })

        # Kiểm tra điều kiện dừng
        if diff < epsilon:
            print(f"Thuật toán hội tụ sau {iteration + 1} lần lặp với |U_n - U_{n-1}| = {diff:.6f}")
            break

        U = new_U

    clusters = np.argmax(U, axis=0)
    return centroids, clusters, iterations

def kmeans_view(request):
    """
    Hàm view xử lý thuật toán K-Means và truyền kết quả về template.
    """
    context = {}
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            k = int(request.POST['k'])
            input_file = request.FILES['input_file']
            points = read_file(input_file)

            # Chạy thuật toán K-Means
            centroids, clusters, iterations = kmeans(points, k)

            # Tạo danh sách chi tiết từng cụm cuối cùng
            final_clusters = []
            for i in range(k):
                cluster_points = points[np.where(clusters == i)]
                final_clusters.append({
                    'cluster_id': i + 1,
                    'centroid': centroids[i],
                    'points': cluster_points.tolist()
                })

            # Vẽ biểu đồ kết quả cuối cùng
            plt.figure(figsize=(8, 6))
            for i in range(k):
                cluster_points = points[clusters == i]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cụm {i+1}', alpha=0.7)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Trọng tâm')
            plt.title("K-Means Clustering - Kết Quả Cuối")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()

            # Chuyển đồ thị thành ảnh base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            graphic = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()

            # Truyền kết quả về context
            context['graphic'] = graphic
            context['final_clusters'] = final_clusters
            context['iterations'] = iterations
            context['result_available'] = True

        except Exception as e:
            context['error'] = f"Lỗi: {e}"

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

def read_file(file):
    """Hàm đọc file bất kỳ: CSV, Excel, TXT."""
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            data = pd.read_excel(file)
        elif file.name.endswith('.txt'):
            data = pd.read_csv(file, delimiter="\t")
        else:
            raise ValueError("File phải có định dạng .csv, .xlsx hoặc .txt")
        return data.values  # Trả về numpy array
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file: {e}")

def kohonen_view(request):
    context = {}
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            rows = int(request.POST['rows'])  # Số dòng (nơron)
            cols = int(request.POST['cols'])  # Số cột (nơron)
            
            # Đọc file dữ liệu từ người dùng
            input_file = request.FILES['input_file']
            data = read_file(input_file)  # Mảng numpy đầu vào

            # Khởi tạo trọng số ngẫu nhiên W (mảng trọng số w_ij)
            wij = np.random.rand(rows, cols, data.shape[1])

            # Tính khoảng cách Euclid từ từng vector đến tất cả nơron
            distance_list = []
            winner_vector = None
            winner_distance = float('inf')
            winner_position = None

            for idx, vector in enumerate(data):
                dist = np.linalg.norm(wij - vector, axis=2)  # Khoảng cách tới tất cả nơron
                min_distance = np.min(dist)  # Khoảng cách nhỏ nhất đến lưới
                winner = np.unravel_index(np.argmin(dist), (rows, cols))  # Nơron chiến thắng

                # Kiểm tra và chọn vector chiến thắng duy nhất
                if min_distance < winner_distance:
                    winner_distance = min_distance
                    winner_vector = vector
                    winner_position = winner

                # Lưu khoảng cách của vector
                distance_list.append({
                    'vector_index': idx + 1,
                    'distances': dist.tolist(),
                    'min_distance': min_distance,
                    'winner': winner
                })

            # Vẽ lưới và đánh dấu nơron chiến thắng duy nhất
            plt.figure(figsize=(8, 8))
            for i in range(rows):
                for j in range(cols):
                    plt.scatter(j, rows - i - 1, s=500, c="lightgray", edgecolors='black')
                    plt.text(j, rows - i - 1, f'({i},{j})', ha='center', va='center', fontsize=8)

            # Đánh dấu nơron chiến thắng
            if winner_position:
                i, j = winner_position
                plt.scatter(j, rows - i - 1, s=500, c="red", edgecolors='black')
                plt.text(j, rows - i - 1, "WIN", ha='center', va='center', color='white', fontsize=10)

            plt.title("Lưới Kohonen và Nơron Chiến Thắng")
            plt.xticks(range(cols))
            plt.yticks(range(rows))
            plt.grid(True)
            plt.gca().invert_yaxis()

            # Chuyển đồ thị thành ảnh base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            graphic = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()

            # Truyền kết quả vào context
            context['distance_list'] = distance_list
            context['winner_vector'] = winner_vector
            context['winner_distance'] = winner_distance
            context['winner_position'] = winner_position
            context['graphic'] = graphic
            context['result_available'] = True

        except Exception as e:
            context['error'] = f"Lỗi: {e}"

    return render(request, 'kohonen.html', context)