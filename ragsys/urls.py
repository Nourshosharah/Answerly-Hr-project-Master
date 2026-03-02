from django.urls import path
from . import views

urlpatterns = [
    path('login/',       views.login_page,  name='login_page'),
    path('login/check/', views.login_check, name='login_check'),

    # Chat sessions
    path('chat/', views.chat_page, name='chat'),
    path('api/chat_eval/new/', views.create_new_chat, name='create_new_chat'),
    path('api/chat_eval/sessions/', views.get_chat_sessions, name='get_chat_sessions'),
    path('api/chat_eval/<str:session_id>/delete/', views.delete_chat_session, name='delete_chat_session'),
    path('api/chat_eval/<str:session_id>/title/', views.update_session_title, name='update_session_title'),


    # path('api/chat/',    views.api_chat,    name='api_chat'),    # ajax
    path('admin-panel/', views.admin_panel, name='admin_panel'),
    path('api/refs/',    views.api_refs,    name='api_refs'), 
    path('logout/',      views.logout_view, name='logout'),
    path('phoenix/',     views.phoenix_embed, name='phoenix'),

    path('data/', views.trigger_data, name='trigger_data'),
    path('data/status/', views.data_status_view, name='data_status'),
    path('data/status/api/', views.data_status_api, name='data_status_api'),

    #path('upload-add/', views.upload_and_add_page, name='upload_and_add_page'),
    #path('api/upload-add/', views.upload_and_add, name='upload_and_add'),
    path('upload-and-add/', views.upload_and_add_page, name='upload_and_add_page'),
    path('api/upload-and-add/', views.upload_and_add, name='upload_and_add'),
    path('api/documents/', views.list_documents, name='list_documents'),
    path('api/documents/delete/', views.delete_document_by_filter, name='delete_document'),

    #eval
    # path('api/chat/', views.api_chat, name='api_chat'),
    path('api/chat_eval/', views.api_chat_with_eval, name='api_chat_with_eval'),
    path('api/evaluate/', views.api_evaluate_only, name='api_evaluate'),
    path('api/trulens/', views.trulens_dashboard, name='trulens_dashboard'),
    path('api/chat_eval/metrics/', views.get_evaluation_metrics, name='get_evaluation_metrics'),


    # ... other paths
    path('api/submit_feedback/', views.submit_rag_feedback, name='submit_feedback'),
]
    
