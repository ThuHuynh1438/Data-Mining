"""
URL configuration for site1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home import views as home
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home.get_home, name='homepage'),
    path('approximation/', home.approximation, name='approximation'),
    path('dependency/', home.dependency, name='dependency'),
    path('reduct/', home.reduct, name='reduct'),
    path('TapPhoBien/', home.TapPhoBien, name='TapPhoBien'),
    path('HSTuongQuan/', home.HSTuongQuan, name='HSTuongQuan'),
    path('gain/', home.decision_tree_gain, name='gain'),
    path('gini/', home.decision_tree_gini, name='gini'),
    path('ifthen/', home.decision_tree_if_then, name='if-then'),
    path('kmeans/', home.kmeans_view, name='kmeans'),
      path('bayes/', home.bayes_view, name='bayes'),
      path('laplace/', home.laplace_view, name='laplace'), 
      path('kohonen/', home.kohonen_view, name='kohonen'), 

    
]
