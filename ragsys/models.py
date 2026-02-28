from django.db import models
from django.contrib.auth.models import User
import uuid
import json

class DisplayMode(models.TextChoices):
    ANSWER_ONLY = 'answer_only', 'Answer only'
    ANSWER_ASSUMPTIONS = 'answer_assumptions', 'Answer + assumptions'
    FULL = 'full', 'Full reasoning'

class ChatSession(models.Model):
    """Stores chat sessions persistently by user"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    title = models.CharField(max_length=200, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    cumulative_tokens = models.IntegerField(default=0)

    # Chat settings
    rag_mode = models.CharField(max_length=100, default="Preprocessed doc")
    temperature = models.FloatField(default=0.7)
    show_thinking = models.BooleanField(default=False)
    
    # Display mode with proper choices
    display_mode = models.CharField(
        max_length=50, 
        default=DisplayMode.ANSWER_ONLY,
        choices=DisplayMode.choices
    )
    
    # Metadata
    message_count = models.IntegerField(default=0)
    last_preview = models.TextField(blank=True, default="")
    
    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['user', '-updated_at']),
            models.Index(fields=['is_active']),
        ]
    
    def __str__(self):
        return f"{self.user.username}: {self.title}"
    
    def add_message(self, role, content, references=None):
        """Helper to add a message with automatic sequencing"""
        message = ChatMessage.objects.create(
            session=self,
            role=role,
            content=content,
            references=json.dumps(references) if references else None,
            sequence=self.message_count
        )
        self.message_count += 1
        self.last_preview = content[:100] + "..." if len(content) > 100 else content
        self.save()
        return message
    
    def get_messages_as_list(self):
        """Get all messages in format suitable for LLM context"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages.all().order_by('sequence')
        ]
    
    def get_display_mode_display(self):
        """Get human-readable display mode"""
        return dict(DisplayMode.choices).get(self.display_mode, self.display_mode)


class ChatMessage(models.Model):
    """Stores individual messages within a chat session"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    sequence = models.IntegerField(default=0)  # Order within session
    
    # For RAG references (stored as JSON string)
    references = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['session', 'sequence']
        unique_together = ['session', 'sequence']  # Prevent duplicate sequences
        indexes = [
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['role']),
        ]
    
    def __str__(self):
        return f"{self.session.title}: {self.role} - {self.content[:50]}..."
    
    def save(self, *args, **kwargs):
        # Auto-set sequence if not provided and this is a new message
        if self.sequence == 0 and not self.pk:
            last_message = ChatMessage.objects.filter(session=self.session).order_by('-sequence').first()
            self.sequence = (last_message.sequence + 1) if last_message else 0
        super().save(*args, **kwargs)
    
    def get_references_json(self):
        """Safely parse references as JSON"""
        if self.references:
            try:
                return json.loads(self.references)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_references(self, ref_data):
        """Set references from a Python object (will be JSON serialized)"""
        if ref_data is not None:
            self.references = json.dumps(ref_data)
        else:
            self.references = None
    
    def has_references(self):
        """Check if message has valid references"""
        return self.references is not None and self.references.strip() != ""
    
    def get_preview(self, length=100):
        """Get a preview of the message content"""
        return self.content[:length] + "..." if len(self.content) > length else self.content