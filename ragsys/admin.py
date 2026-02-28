# ragsys/admin.py
from django.contrib import admin
from .models import ChatSession, ChatMessage

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'message_count', 'created_at', 'updated_at']
    list_filter = ['user', 'created_at', 'rag_mode']
    search_fields = ['title', 'user__username']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'role', 'sequence', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['content', 'session__title']
    readonly_fields = ['created_at']