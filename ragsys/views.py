####new
from django.shortcuts import render,redirect
# Create your views here.
import json, os, pathlib, subprocess, threading,time,re
from django.http import JsonResponse, HttpResponseRedirect, HttpResponse
from utils.upload_data_manually import upload_data_manually
from utils.chatbot import ChatBot   # your existing file
from langchain_chroma import Chroma
import threading
import queue
import logging
import io
from django.urls import reverse
from django.views.decorators.http import require_http_methods
import json
from utils.chatbot import ChatBot
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required  # optional, if using Django auth
import os
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from utils.load_config import LoadConfig
from datetime import datetime
from django.utils import timezone
from django.contrib.auth.models import User
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from .models import ChatSession, ChatMessage
from django.utils.dateparse import parse_datetime
from .ldap_auth import LDAPAuth  # Import LDAP auth
import json
import time
from django.http import JsonResponse
from django.shortcuts import render, HttpResponseRedirect
import threading
from django.core.cache import cache
from utils.trulens_evaluator import TruLensEvaluator, RAGWithTruLens
from utils.chatbot import ChatBot
#  Add these imports at the top of views.py if not already present
import json
from datetime import datetime
from django.utils import timezone
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import requests
import logging
# from .models import Feedback, ChatSession, ChatMessage

forbidden_phrases = ["إن المعلومات المذكورة هنا هي ملكية خاصة لشركة سيريتل، ويجب ألاّ يتم استخدامها أو نسخها أو إظهارها إلا بتصريح مكتوب من المالك. يتعهد مستلم هذه المعلومات بالاحتفاظ بها واستخدامها، كما يوافق على حمايتها من الضياع، السرقة أو الاستخدام غير المسموح به.", 
                     "THE INFORMATION CONTAINED HEREIN IS PROPRIETARY TO SYRIATEL, AND IT SHALL NOT BE USED, REPRODUCED\nOR DISCLOSED TO OTHERS EXCEPT AS SPECIFICALLY PERMITTED IN WRITING BY THE PROPRIETOR. THE RECIPIENT\nOF THIS INFORMATION, BY ITS RETENTION AND USE, AGREES TO PROTECT THE SAME FROM LOSS, THEFT OR\nUNAUTHORIZED USE.",
                     "the information contained herein is proprietary to syriatel, and it shall not be used, reproduced\nor disclosed to others except as specifically permitted in writing by the proprietor. the recipient\nof this information, by its retention and use, agrees to protect the same from loss, theft or\nunauthorized use."]


session_idd=0

def chat_page(request):
    if not request.session.get('user'):
        return HttpResponseRedirect('/login/')
    # Initialize session storage
    init_user_session_storage(request)
    # Get session ID from query parameter
    session_id = request.GET.get('session')
    # If session ID provided and user is authenticated, try to load from database
    user = request.user if request.user.is_authenticated else None
    current_session_data = None
    if session_id and user:
        try:
            db_session = ChatSession.objects.get(id=session_id, user=user)
            session_id = str(db_session.id)
            current_session_data = {
                'id': session_id,
                'title': db_session.title,
                'history': get_session_history_from_db(db_session),
                'rag_mode': db_session.rag_mode,
                'temperature': db_session.temperature,
                'display_mode': db_session.display_mode
            }
            request.session['active_session'] = session_id
        except (ChatSession.DoesNotExist, ValueError):
            pass
    # If no valid session from database, use session storage
    if not current_session_data:
        if session_id:
            if session_id in request.session.get('chat_sessions', {}):
                request.session['active_session'] = session_id
            else:
                session_id = create_session_combined(request)[0]
        if not request.session.get('active_session'):
            session_id = create_session_combined(request)[0]
        else:
            session_id = request.session['active_session']
        # Get current session data from storage
        session_data = get_session_from_storage(request, session_id) or {}
        current_session_data = {
            'id': session_id,
            'title': session_data.get('title', 'New Chat'),
            'history': session_data.get('history', []),
            'rag_mode': session_data.get('rag_mode', 'Preprocessed doc'),
            'temperature': session_data.get('temperature', 0.7),
            'display_mode': session_data.get('display_mode', 'answer_only')
        }
    # Get all sessions for sidebar (combined from database and storage)
    sessions_list = []
    # Get from database if user is authenticated
    if user:
        db_sessions = ChatSession.objects.filter(user=user).order_by('-updated_at')[:50]
        for session in db_sessions:
            sessions_list.append({
                'id': str(session.id),
                'title': session.title,
                'is_active': str(session.id) == session_id,
                'preview': session.last_preview[:50] + '...' if session.last_preview else 'Empty chat',
                'created_at': session.created_at,
                'message_count': session.message_count,
                'last_activity': session.updated_at,
                'from_db': True
            })
    # Also include session storage sessions
    storage_sessions = get_all_sessions_from_storage(request)
    for storage_session in storage_sessions:
        # Skip if already in database list
        if not any(s.get('id') == storage_session['id'] for s in sessions_list):
            sessions_list.append(storage_session)
    # Sort by last activity - FIXED VERSION
    def get_sort_key(session_item):
        last_activity = session_item.get('last_activity', '')
        if hasattr(last_activity, 'isoformat'):  # It's a datetime object
            return last_activity
        elif isinstance(last_activity, str):
            try:
                # Try to parse ISO format string
                parsed = parse_datetime(last_activity)
                if parsed:
                    return parsed
            except:
                pass
        # Return empty string as fallback
        return ''
    sessions_list.sort(key=get_sort_key, reverse=True)
    return render(request, 'chat.html', {
        'user': request.session['user'],
        'current_session': current_session_data,
        'all_sessions': sessions_list,
        'active_session_id': session_id,
        'trulens_available': True
    })

def build_flexible_pattern(text: str):
    # Normalize: collapse all whitespace (including \n) into single spaces, then split into words
    words = re.split(r'\s+', text.strip())
    escaped_words = [re.escape(word) for word in words if word]
    pattern = r'\s+'.join(escaped_words)
    return re.compile(pattern, re.IGNORECASE)

# Build patterns ONCE (e.g., at module level or in __init__)
forbidden_patterns = [build_flexible_pattern(phrase) for phrase in forbidden_phrases]

# # ---------- hard-coded users ----------
# USERS = {
#     'hr':  {'email':'hr@sy',  'role':'hr'},
#     'emp': {'email':'emp@sy', 'role':'emp'}
# }
# ---------- hard-coded users ----------
USERS = {
    "dana_hosh": {"email": "dana_hosh@sy", "role": "emp"},
    "waseem_kattan": {"email": "waseem_kattan@sy", "role": "emp"},
    "mhd_hussein_nahhas": {"email": "hussein_nahhas@sy", "role": "emp"},
    "haitham_jameh": {"email": "haitham_jameh@sy", "role": "emp"},
    "elias_issawi": {"email": "elias_issawi@sy", "role": "emp"},
    "ashtar_kassis": {"email": "ashtar_kassis@sy", "role": "emp"},
    "rouba_kokash": {"email": "rouba_kokash@sy", "role": "emp"},
    "manal_jbawi": {"email": "manal_jbawi@sy", "role": "emp"},
    "mhd_wafik_ismail": {"email": "wafik_ismail@sy", "role": "emp"},
    "alaa_wannous": {"email": "alaa_wannous@sy", "role": "emp"},
    "samara_ali": {"email": "samara_ali@sy", "role": "emp"},
    "rashad_fatayri": {"email": "rashad_fatayri@sy", "role": "emp"},
    "lama_mansour": {"email": "lama_mansour@sy", "role": "emp"},
    "haneen_abouleil": {"email": "haneen_abouleil@sy", "role": "emp"},
    "ayat_abdulaziz": {"email": "ayat_abdulaziz@sy", "role": "emp"},
    "dania_awad": {"email": "dania_awad@sy", "role": "emp"},
    "hussein_khateeb": {"email": "hussein_khateeb@sy", "role": "emp"},
    "khitam_barhoum": {"email": "khitam_barhoum@sy", "role": "emp"},
    "maya_amina": {"email": "maya_amina@sy", "role": "emp"},
    "hala_hariri": {"email": "hala_hariri@sy", "role": "emp"},
    "diaaeddin_khalifa": {"email": "diaaeddin_khalifa@sy", "role": "emp"},
    "bassam_murad": {"email": "bassam_murad@sy", "role": "emp"},
    "ammar_mahfoud": {"email": "ammar_mahfoud@sy", "role": "emp"},
    "linda_saleh": {"email": "linda_saleh@sy", "role": "emp"},
    "suzan_khateeb": {"email": "suzan_khateeb@sy", "role": "emp"},
    "ossama_hasan": {"email": "ossama_hasan@sy", "role": "emp"},
    "alaa_katalan": {"email": "alaa_katalan@sy", "role": "emp"},
    "maya_fandi": {"email": "maya.fandi@sy", "role": "emp"},
    "shaza_boudi": {"email": "shaza_boudi@sy", "role": "emp"},
    "taher_barakat": {"email": "taher_barakat@sy", "role": "emp"},
    "duaa_aadleh": {"email": "duaa_aadleh@sy", "role": "emp"},
    "salem_ahmad": {"email": "salem_ahmad@sy", "role": "emp"},
    "rania_samour": {"email": "rania_samour@sy", "role": "emp"},
    "alaa_faaour": {"email": "alaa_faaour@sy", "role": "emp"},
    "majd_aboud": {"email": "majd_aboud@sy", "role": "emp"},
    "mhd_soulaiman": {"email": "mhd_soulaiman@sy", "role": "emp"},
    "manal_kurdi": {"email": "manal_kurdi@sy", "role": "emp"},
    "alaa_aaji": {"email": "alaa_aaji@sy", "role": "emp"},
    "heba_ayoub": {"email": "heba_ayoub@sy", "role": "emp"},
    "nour_shoshara": {"email": "nour_shoshara@sy", "role": "emp"},
    "tasneem_mokhanek": {"email": "tasneem_mokhanek@sy", "role": "emp"},
    "maher_rashid": {"email": "maher_rashid@sy", "role": "emp"},
    'hr':  {'email':'hr@sy',  'role':'hr'},
    'emp': {'email':'emp@sy', 'role':'emp'},
    'rama_sous':{'email':'rama_sous@sy', 'role':'emp'},
    'loay_salloum':{'email':'loay_salloum@sy', 'role':'emp'},
    'mahdi_othman':{'email':'mahdi_othman@sy', 'role':'emp'},
    'najm_saleh':{'email':'najm_saleh@sy', 'role':'emp'},
         'sami_aboubala':{'email':'sami_aboubala@sy', 'role':'emp'},
         'riham_balhas':{'email':'riham_balhas@sy', 'role':'emp'},
          'zein_mubarak':{'email':'zein_mubarak@sy', 'role':'emp'},
    
    'mhd_farra':{'email':'mhd_farra@sy', 'role':'emp'},
    "nakhleh_kallas": {"email": "nakhleh.kallas@sy", "role": "hr"},
    "marah_daher": {"email": "marah.daher@sy", "role": "hr"},
    "maram_soulaiman": {"email": "maram.soulaiman@sy", "role": "hr"},
    "raghed_mashreke": {"email": "raghed.mashreke@sy", "role": "hr"},
    "meriam_haddad": {"email": "meriam.haddad@sy", "role": "hr"},
    "raghad_tabaa": {"email": "raghad.tabaa@sy", "role": "hr"},
    "rawan_akl": {"email": "rawan.akl@sy", "role": "hr"},
    "rita_habeeb": {"email": "rita.habeeb@sy", "role": "hr"}

}

# ---------- helpers ----------
def _in_role(request, role):
    u = request.session.get('user')
    return u and u.get('role') == role

# ---------- 1. LOGIN ----------
def login_page(request):
    return render(request, 'login.html')

def login_check(request):
    if request.method != 'POST':
        return JsonResponse({'ok': False})
    body = json.loads(request.body)
    username = body.get('username')
    email = body.get('email')
    password = body.get('password', '')
    # Try LDAP authentication if password is provided
    if password:
        ldap_auth = LDAPAuth()
        ldap_user = ldap_auth.authenticate(username, password)
        if ldap_user:
            # Determine role from LDAP
            role = determine_role_from_ldap(ldap_user)
            # Use email from LDAP if available, otherwise use provided email
            user_email = ldap_user.get('email', email)
            # Get or create Django user
            user = get_or_create_django_user(
                username=username,
                email=user_email,
                role=role
            )
            # Log the user in using Django's auth system
            user.backend = 'django.contrib.auth.backends.ModelBackend'
            auth_login(request, user)
            # Store in session for compatibility
            request.session['user'] = {
                'name': username,
                'email': user_email,
                'role': role,
                'ldap_user': True,
                'first_name': ldap_user.get('first_name', ''),
                'last_name': ldap_user.get('last_name', '')
            }
            # Initialize session storage for chat sessions
            init_user_session_storage(request)
            # Check if user has any sessions
            session_count = get_user_session_count(user, request)
            # Only create new session if user has no existing sessions
            if session_count == 0:
                create_session_combined(request)
            else:
                # Set active session to the most recent one
                if user.is_authenticated:
                    latest_session = ChatSession.objects.filter(user=user).order_by('-updated_at').first()
                    if latest_session:
                        request.session['active_session'] = str(latest_session.id)
                elif request.session.get('chat_sessions'):
                    # Get latest from storage
                    sessions = get_all_sessions_from_storage(request)
                    if sessions:
                        request.session['active_session'] = sessions[0]['id']
            return JsonResponse({'ok': True, 'next': '/chat/'})
    # Fallback to hardcoded users (without password check)
    if username in USERS and USERS[username]['email'] == email:
        # Get or create Django user
        user = get_or_create_django_user(username, email, USERS[username]['role'])
        # Log the user in using Django's auth system
        user.backend = 'django.contrib.auth.backends.ModelBackend'
        auth_login(request, user)
        # Also store in session for compatibility
        request.session['user'] = {
            'name': username,
            'email': email,
            'role': USERS[username]['role'],
            'ldap_user': False
        }
        # Initialize session storage for chat sessions
        init_user_session_storage(request)
        # Check if user has any sessions
        session_count = get_user_session_count(user, request)
        # Only create new session if user has no existing sessions
        if session_count == 0:
            create_session_combined(request)
        else:
            # Set active session to the most recent one
            if user.is_authenticated:
                latest_session = ChatSession.objects.filter(user=user).order_by('-updated_at').first()
                if latest_session:
                    request.session['active_session'] = str(latest_session.id)
            elif request.session.get('chat_sessions'):
                # Get latest from storage
                sessions = get_all_sessions_from_storage(request)
                if sessions:
                    request.session['active_session'] = sessions[0]['id']
        return JsonResponse({'ok': True, 'next': '/chat/'})
    return JsonResponse({'ok': False, 'error': 'Invalid credentials'})

@require_http_methods(["POST"])
def create_new_chat(request):
    if not request.session.get('user'):
        return JsonResponse({'error': 'login required'}, status=403)
    try:
        body = json.loads(request.body) if request.body else {}
    except:
        body = {}
    # Generate title from first message if provided
    first_message = body.get('first_message', '').strip()
    if first_message:
        title = first_message[:30] + ('...' if len(first_message) > 30 else '')
    else:
        title = "New Chat"
    # Get settings from request or use defaults
    rag_mode = 'Preprocessed doc'
    temperature =  0.1
    display_mode = body.get('display_mode', 'answer_only')
    # Create session in both storage and database
    session_id, db_session = create_session_combined(
        request,
        title=title,
        rag_mode=rag_mode,
        temperature=temperature,
        display_mode=display_mode,
        first_message=first_message
    )
    return JsonResponse({
        'success': True,
        'session_id': session_id,
        'title': title,
        'redirect_url': f'/chat/?session={session_id}'
    })

# --------------------------------------------------
# API: Get all chat sessions for sidebar
# --------------------------------------------------
@require_http_methods(["GET"])
def get_chat_sessions(request):
    if not request.session.get('user'):
        return JsonResponse({'error': 'login required'}, status=403)
    sessions_list = []
    active_session = request.session.get('active_session')
    # Get from database if user is authenticated
    user = request.user if request.user.is_authenticated else None
    if user:
        db_sessions = ChatSession.objects.filter(user=user).order_by('-updated_at')[:50]
        for session in db_sessions:
            sessions_list.append({
                'id': str(session.id),
                'title': session.title,
                'is_active': str(session.id) == active_session,
                'preview': session.last_preview[:50] + '...' if session.last_preview else 'Empty chat',
                'created_at': session.created_at.isoformat(),
                'message_count': session.message_count,
                'from_db': True
            })
    # Also include session storage sessions
    storage_sessions = get_all_sessions_from_storage(request)
    for storage_session in storage_sessions:
        if not any(s.get('id') == storage_session['id'] for s in sessions_list):
            sessions_list.append(storage_session)
    return JsonResponse({
        'sessions': sessions_list,
        'active_session': active_session
    })

# --------------------------------------------------
# API: Update session title
# --------------------------------------------------
@require_http_methods(["PUT"])
def update_session_title(request, session_id):
    if not request.session.get('user'):
        return JsonResponse({'error': 'login required'}, status=403)
    try:
        body = json.loads(request.body)
        new_title = body.get('title', '').strip()
    except:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    if not new_title:
        return JsonResponse({'error': 'Title required'}, status=400)
    # Update in database if user is authenticated
    user = request.user if request.user.is_authenticated else None
    if user:
        try:
            db_session = ChatSession.objects.get(id=session_id, user=user)
            db_session.title = new_title[:200]
            db_session.save()
        except (ChatSession.DoesNotExist, ValueError):
            pass
    # Update in session storage
    success = update_session_in_storage(request, session_id, {
        'title': new_title,
        'title_set': True
    })
    if success:
        return JsonResponse({'success': True})
    return JsonResponse({'error': 'Session not found'}, status=404)

# --------------------------------------------------
# API: Delete session
# --------------------------------------------------
@require_http_methods(["DELETE"])
def delete_chat_session(request, session_id):
    if not request.session.get('user'):
        return JsonResponse({'error': 'login required'}, status=403)
    # Delete from database if user is authenticated
    user = request.user if request.user.is_authenticated else None
    if user:
        try:
            ChatSession.objects.filter(id=session_id, user=user).delete()
        except:
            pass
    # Handle session storage
    chat_sessions = request.session.get('chat_sessions', {})
    if session_id in chat_sessions:
        # If this is the last session, clear it instead of deleting
        if len(chat_sessions) == 1:
            update_session_in_storage(request, session_id, {
                'title': 'New Chat',
                'history': [],
                'last_refs': '',
                'title_set': False,
                'rag_mode': 'Preprocessed doc',
                'temperature': 0.7,
                'display_mode': 'answer_only'
            })
            return JsonResponse({'success': True, 'cleared': True})
        else:
            # If there are multiple sessions, delete this one
            delete_session_from_storage(request, session_id)
            return JsonResponse({'success': True})
    return JsonResponse({'error': 'Session not found'}, status=404)

def admin_panel(request):
    if not _in_role(request, 'hr'):
        return HttpResponse('<div class="bidi-text">FORBIDDEN</div>', status=403)
    stats = {
        'total_chats': 42,      # dummy – replace with real counters
        'total_messages': 420,
        'active_user': request.session['user']['name']
    }
    return render(request, 'admin_panel.html', stats)

# Your existing ChatBot
# Initialize TruLens evaluator (do once at module load)
TRULENS_EVALUATOR = TruLensEvaluator(
    vllm_base_url="http://localhost:8001",  # Your vLLM endpoint
    vllm_model_name=None,  # Auto-detect from server
    database_url="sqlite:///trulens_eval.db"
)

def api_chat_with_eval(request):
    """
    Chat API that returns response immediately, evaluates in background.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    if not request.session.get('user'):
        return JsonResponse({'error': 'login required'}, status=403)
    # Get active session
    session_id = request.session.get('active_session')
    if not session_id:
        session_id = create_session_combined(request)[0]
    session_data = get_session_from_storage(request, session_id)
    if not session_data:
        session_id = create_session_combined(request)[0]
        session_data = get_session_from_storage(request, session_id)
    body = json.loads(request.body)
    msg = body.get('message', '').strip()
    dtype = 'Preprocessed doc'
    temp = 0.1
    display_mode = body.get('display_mode', 'answer_only')
    if display_mode not in ["answer_only", "answer_assumptions", "full"]:
        display_mode = "answer_only"
    # Update session settings
    update_data = {
        'rag_mode': 'Preprocessed doc',
        'temperature': 0.1,
        'display_mode': display_mode
    }
    # Update session title if first message
    if not session_data.get('title_set', False) and msg:
        title = msg[:30] + ('...' if len(msg) > 30 else '')
        update_data['title'] = title
        update_data['title_set'] = True
        # Update in database too
        user = request.user if request.user.is_authenticated else None
        if user:
            try:
                db_session = ChatSession.objects.get(id=session_id, user=user)
                db_session.title = title[:200]
                db_session.save()
            except (ChatSession.DoesNotExist, ValueError):
                pass
    update_session_in_storage(request, session_id, update_data)
    # Add user message
    add_message_combined(session_id, request, 'user', msg)
    # ========== STEP 1: Get bot response (measure latency) ==========
    start_time = time.time()
    _, updated_history, refs, prompt_tk, completion_tk, span_id, session_id = ChatBot.respond(
        chatbot=[],
        message=msg,
        data_type=dtype,
        temperature=temp,
        display_mode=display_mode,
        session_id=session_id,
        user_id=request.session.get('user', {}).get('name') if request.user.is_authenticated else None,
    )
    session_idd = session_id
    latency_sec = round(time.time() - start_time, 2)
    total_tokens = prompt_tk + completion_tk
    # Calculate cost
    cost_per_1m_input = 0.50
    cost_per_1m_output = 1.50
    estimated_cost = (
        (prompt_tk / 1_000_000) * cost_per_1m_input +
        (completion_tk / 1_000_000) * cost_per_1m_output
    )
    estimated_cost = round(estimated_cost, 6)
    bot_response = updated_history[-1][1] if updated_history and len(updated_history[-1]) > 1 else "" 
    
    # bot_response = updated_history
    # Add bot response to storage with token count
    add_message_combined(session_id, request, 'assistant', bot_response, refs, tokens_used=total_tokens)
    # Update session storage
    update_session_in_storage(request, session_id, {
        'history': updated_history[-10:],
        'last_refs': refs
    })
    user_html = f'<div class="msg user">{msg}</div>'
    bot_html = f'<div class="msg bot">{bot_response}</div>'
    # ========== STEP 2: Start background evaluation ==========
    # Generate unique evaluation key for this response
    eval_key = f"eval_metrics_{session_id}_{int(time.time())}"
    
    def run_evaluation_background():
        """Run TruLens evaluation in background thread"""
        try:
            eval_start = time.time()
            metrics = TRULENS_EVALUATOR.evaluate_rag_response(
                question=msg,
                response=bot_response,
                context=refs,
            )
            eval_time = round(time.time() - eval_start, 2)
            # Store metrics in cache with 5-minute expiry
            cache.set(eval_key, {
                'metrics': metrics,
                'eval_time_sec': eval_time,
                'completed_at': time.time()
            }, timeout=300)  # 5 minutes
            print(f"? Evaluation completed in {eval_time}s, stored under key: {metrics}")
            annotation_payload = {"data": []}
            for metric_name, value in metrics.items():
                print(f"metric_name :::   {metric_name}")
                annotation_payload["data"].append({
                    "span_id": span_id,  # ? span_id ?? rag_pipeline
                    "name": f"eval.{metric_name}",
                    "annotator_kind": "LLM",
                    "result": {
                        "score": float(value),
                        "label": "high" if value >= 0.8 else "medium" if value >= 0.6 else "low",
                        "explanation": f"TruLens {metric_name} evaluation"
                    },
                    "metadata": {
                        "session_id": session_idd,
                        # "user_id": user_id,
                        "evaluator": "trulens",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "identifier": f"{session_id}_{metric_name}_{int(time.time())}"
                })
            # ????? ??? Phoenix
            requests.post(
                "http://localhost:6006/v1/span_annotations?sync=false",
                json=annotation_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            print(f"? Evaluation sent to Phoenix for span {span_id} {annotation_payload}")
        except Exception as e:
            print(f"? Evaluation error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Store error state
            cache.set(eval_key, {
                'error': str(e),
                'completed_at': time.time()
            }, timeout=300)
    
    # Start background thread for evaluation
    eval_thread = threading.Thread(target=run_evaluation_background, daemon=True)
    eval_thread.start()
    # ========== STEP 3: Get cumulative tokens from DB ==========
    cumulative_tokens = 0
    user = request.user if request.user.is_authenticated else None
    if user:
        try:
            db_session = ChatSession.objects.get(id=session_id, user=user)
            cumulative_tokens = db_session.cumulative_tokens
        except (ChatSession.DoesNotExist, ValueError):
            pass
    # ========== STEP 4: Return immediate response ==========
    return JsonResponse({
        'user_html': user_html,
        'bot_html': bot_html,
        'refs_html': refs,
        'latency_sec': latency_sec,
        'total_tokens': total_tokens,
        'cumulative_tokens': cumulative_tokens,  # NEW: Cumulative tokens for session
        'estimated_cost_usd': round(estimated_cost, 6),
        'session_id': session_id,
        'span_id': span_id,
        'eval_key': eval_key,  # Frontend will use this to poll for metrics
        'evaluation_status': 'running'  # Indicates evaluation is in progress
    })

@require_http_methods(["GET"])
def get_evaluation_metrics(request):
    """
    Poll endpoint to get evaluation metrics after they complete.
    Frontend should poll this endpoint using the eval_key from chat response.
    """
    if not request.session.get('user'):
        return JsonResponse({'error': 'login required'}, status=403)
    eval_key = request.GET.get('eval_key')
    if not eval_key:
        return JsonResponse({'error': 'eval_key required'}, status=400)
    # Try to get metrics from cache
    metrics_data = cache.get(eval_key)
    if metrics_data is None:
        # Still running or not started
        return JsonResponse({
            'status': 'pending',
            'message': 'Evaluation still in progress'
        })
    if 'error' in metrics_data:
        # Evaluation failed
        return JsonResponse({
            'status': 'error',
            'error': metrics_data['error']
        })
    print("metrics to phoenix___________________________",metrics_data)
    # Evaluation completed successfully
    return JsonResponse({
        'status': 'completed',
        'metrics': metrics_data['metrics'],
        'eval_time_sec': metrics_data['eval_time_sec'],
        'trulens': {
            'status': 'success',
            'metrics': metrics_data['metrics']
        }
    })

# API: Reset user sessions (Admin/HR only)
# --------------------------------------------------
@require_http_methods(["POST"])
def reset_user_sessions_api(request, username):
    """
    API endpoint to reset all sessions for a specific user.
    Only accessible to HR/admin users.
    """
    if not _in_role(request, 'hr'):
        return JsonResponse({'error': 'HR role required'}, status=403)
    try:
        # Get the target user
        target_user = User.objects.get(username=username)
        # Can't reset HR user sessions if not superuser
        if target_user.username == 'hr' and not request.user.is_superuser:
            return JsonResponse({'error': 'Cannot reset HR user sessions without superuser privileges'}, status=403)
        # Reset the sessions
        success, message = reset_user_sessions(target_user)
        if success:
            # Also clear session storage if this is the current user
            current_username = request.session.get('user', {}).get('name')
            if current_username == username:
                # Clear session storage for current request
                if 'chat_sessions' in request.session:
                    del request.session['chat_sessions']
                if 'active_session' in request.session:
                    del request.session['active_session']
                # Re-initialize
                init_user_session_storage(request)
                create_session_combined(request)
            return JsonResponse({
                'success': True,
                'message': message,
                'username': username
            })
        else:
            return JsonResponse({'error': message}, status=400)
    except User.DoesNotExist:
        return JsonResponse({'error': f'User {username} not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def api_evaluate_only(request):
    """
    Standalone evaluation endpoint - evaluate a previous response.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    body = json.loads(request.body)
    question = body.get('question', '')
    response = body.get('response', '')
    context = body.get('context', '')
    metrics_to_run = body.get('metrics', 'all')  # 'all', 'rag_triad', or list
    if not all([question, response]):
        return JsonResponse({'error': 'question and response required'}, status=400)
    run_all = metrics_to_run == 'all'
    start_time = time.time()
    metrics = TRULENS_EVALUATOR.evaluate_rag_response(
        question=question,
        response=response,
        context=context or "",
        # run_all_metrics=run_all
    )
    eval_time = round(time.time() - start_time, 2)
    return JsonResponse({
        'metrics': metrics,
        'eval_time_sec': eval_time
    })

def trulens_dashboard(request):
    """
    Redirect to TruLens dashboard or return status.
    """
    try:
        from trulens.dashboard import run_dashboard
        # Note: This starts a separate Streamlit server
        # In production, you'd run this as a separate service
        url = run_dashboard(TRULENS_EVALUATOR.session, port=8502, _dev=False)
        return JsonResponse({'dashboard_url': str(url)})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

APPCFG = LoadConfig()
# # forbidden_phrases = ["إن المعلومات المذكورة هنا هي ملكية خاصة لشركة سيريتل، ويجب ألاّ يتم استخدامها أو نسخها أو إظهارها إلا بتصريح مكتوب من المالك. يتعهد مستلم هذه المعلومات بالاحتماظ بها واستخدامها، كما يواقي على حمايتها من الضياع، السرقة أو الاستخدام غير المسموح به.","THE INFORMATION CONTAINED HEREIN IS PROPRIETARY TO SYRIATEL, AND IT SHALL NOT BE USED, REPRODUCED OR DISCLOSED TO OTHERS EXCEPT AS SPECIFICALLY PERMITTED IN WRITING BY THE PROPRIETOR. THE RECIPIENT
# OF THIS INFORMATION, BY ITS RETENTION AND USE, AGREES TO PROTECT THE SAME FROM LOSS, THEFT OR
# UNAUTHORIZED USE.",
# "the information contained herein is proprietary to syriatel, and it shall not be used, reproduced
# or disclosed to others except as specifically permitted in writing by the proprietor. the recipient
# of this information, by its retention and use, agrees to protect the same from loss, theft or
# unauthorized use."]

# # -------------------------
# # Page View (with HR check)
# # -------------------------
# def upload_and_add_page(request):
#     user = request.session.get('user')
#     if not user:
#         return redirect('/login/')
#     if user.get('role') != 'hr':
#         return render(request, 'error.html', {
#             'error': 'Access denied. HR role required.'
#         }, status=403)
#     return render(request, 'upload_and_add.html')

# # -------------------------
# # API View (with HR check)
# # -------------------------
# @require_http_methods(["POST"])
# def upload_and_add(request):
#     user = request.session.get('user')
#     if not user:
#         return JsonResponse({'error': 'Login required'}, status=403)
#     if user.get('role') != 'hr':
#         return JsonResponse({'error': 'HR role required to add documents.'}, status=403)
#     pdf_file = request.FILES.get('pdf_file')
#     if not pdf_file or not pdf_file.name.endswith('.pdf'):
#         return JsonResponse({'error': 'Please upload a valid PDF file.'}, status=400)
#     # Ensure the target directory exists
#     os.makedirs(APPCFG.data_directory, exist_ok=True)
#     try:
#         # --- STEP 1: Save original file to APPCFG.data_directory ---
#         safe_filename = os.path.basename(pdf_file.name)
#         # Optional: prevent path traversal
#         safe_filename = "".join(c for c in safe_filename if c.isalnum() or c in "._- ")
#         save_path = os.path.join(APPCFG.data_directory, safe_filename)
#         # Avoid overwriting: add suffix if file exists
#         counter = 1
#         base, ext = os.path.splitext(safe_filename)
#         while os.path.exists(save_path):
#             safe_filename = f"{base}_{counter}{ext}"
#             save_path = os.path.join(APPCFG.data_directory, safe_filename)
#             counter += 1
#         with open(save_path, 'wb+') as destination:
#             for chunk in pdf_file.chunks():
#                 destination.write(chunk)
#         print(f"Saved uploaded PDF to: {save_path}")
#         # --- STEP 2: Load and preprocess from the SAVED file ---
#         loader = PyPDFLoader(save_path)
#         raw_pages = loader.load()
#         # Update metadata 'source' to point to the saved file (not temp)
#         for doc in raw_pages:
#             doc.metadata['source'] = save_path  # critical for reference links
#         # --- STEP 3: Preprocessing (same as before) ---
#         cleaned_pages = []
#         for doc in raw_pages:
#             new_content = doc.page_content
#             removed_any = False
#             for pattern in forbidden_patterns:
#                 while pattern.search(new_content):
#                     removed_any = True
#                     new_content = pattern.sub('', new_content)
#             if removed_any:
#                 import re
#                 new_content = re.sub(r'\n\s*\n\s*\n', '\n', new_content)
#                 new_content = re.sub(r' +', ' ', new_content).strip()
#             cleaned_pages.append(Document(page_content=new_content, metadata=doc.metadata))
#         # Drop first page
#         if cleaned_pages:
#             cleaned_pages = cleaned_pages[1:]
#         # Cut from 2nd "Related Documents"
#         cut_from = None
#         count = 0
#         trigger_phrases = ["related documents", "الوثائق المتعلقة"]
#         for idx, doc in enumerate(cleaned_pages):
#             if any(trigger in doc.page_content.lower() for trigger in trigger_phrases):
#                 count += 1
#                 if count == 2:
#                     cut_from = idx
#                     break
#         if cut_from is not None:
#             cleaned_pages = cleaned_pages[:cut_from]
#         # --- STEP 4: Add to existing vector DB ---
#         if not os.path.exists(APPCFG.persist_directory):
#             return JsonResponse({'error': 'VectorDB not initialized. Run initial ingestion first.'}, status=400)
#         vectordb = Chroma(
#             persist_directory=APPCFG.persist_directory,
#             embedding_function=APPCFG.embedding_model,
#         )
#         non_empty = [doc for doc in cleaned_pages if doc.page_content.strip()]
#         if not non_empty:
#             return JsonResponse({'error': 'No valid content after cleaning.'}, status=400)
#         ids = [str(uuid.uuid4()) for _ in non_empty]
#         vectordb.add_documents(documents=non_empty, ids=ids)
#         vectordb.persist()
#         return JsonResponse({
#             'success': True,
#             'added_docs': len(non_empty),
#             'saved_file': safe_filename
#         })
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return JsonResponse({'error': str(e)}, status=500)

# ---------- 4. DATA TRIGGER ----------
# views.py
import queue, threading
_done_queue = queue.Queue()
# Global state (use cache/database in production)
LOG_BUFFER = io.StringIO()
LOG_LOCK = threading.Lock()
TASK_STATUS = {'status': None, 'docs_indexed': 0}  # 'pending', 'running', 'ok', 'error'
_done_queue = queue.Queue()

# Custom logging handler to capture logs
class BufferHandler(logging.Handler):
    def emit(self, record):
        with LOG_LOCK:
            LOG_BUFFER.write(self.format(record) + '\n')

# Set up logger
logger = logging.getLogger('data_upload')
logger.setLevel(logging.INFO)
logger.handlers.clear()
buffer_handler = BufferHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
buffer_handler.setFormatter(formatter)
logger.addHandler(buffer_handler)

def _wrapped_upload():
    global TASK_STATUS
    try:
        logger.info("Starting manual data upload...")
        count = upload_data_manually()  # should return int
        logger.info(f"Upload completed successfully. Indexed {count} documents.")
        TASK_STATUS = {'status': 'ok', 'docs_indexed': count}
        _done_queue.put(('ok', count))
    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Upload failed: {error_msg}")
        TASK_STATUS = {'status': 'error', 'docs_indexed': 0}
        _done_queue.put(('error', error_msg))

def trigger_data(request):
    if not _in_role(request, 'hr'):
        return HttpResponse('Forbidden', status=403)
    # Reset global state before new run
    global LOG_BUFFER, TASK_STATUS
    with LOG_LOCK:
        LOG_BUFFER = io.StringIO()
        TASK_STATUS = {'status': 'pending', 'docs_indexed': 0}
    # Start background thread
    threading.Thread(target=_wrapped_upload, daemon=True).start()
    # Redirect to a status page that uses AJAX
    return redirect('data_status')

def data_status_view(request):
    if not _in_role(request, 'hr'):
        return HttpResponse('Forbidden', status=403)
    return render(request, 'data_status.html')

@require_http_methods(["GET"])
def data_status_api(request):
    if not _in_role(request, 'hr'):
        return JsonResponse({'error': 'Forbidden'}, status=403)
    # Get current logs
    with LOG_LOCK:
        logs = LOG_BUFFER.getvalue()
    return JsonResponse({
        'status': TASK_STATUS['status'],
        'docs_indexed': TASK_STATUS['docs_indexed'],
        'logs': logs,
    })

def api_refs(request):
    """Return the last cleaned references (markdown)."""
    if not request.session.get('user'):
        return JsonResponse({'error': 'login required'}, status=403)
    session_id = request.session.get('active_session')
    if not session_id:
        return JsonResponse({'refs_html': ''})
    # Try to get from database first
    user = request.user if request.user.is_authenticated else None
    if user:
        try:
            db_session = ChatSession.objects.get(id=session_id, user=user)
            last_message = ChatMessage.objects.filter(
                session=db_session,
                references__isnull=False
            ).order_by('-sequence').first()
            if last_message:
                return JsonResponse({'refs_html': last_message.references})
        except (ChatSession.DoesNotExist, ValueError):
            pass
    # Fallback to session storage
    session_data = get_session_from_storage(request, session_id)
    if session_data:
        refs = session_data.get('last_refs', '')
        return JsonResponse({'refs_html': refs})
    return JsonResponse({'refs_html': ''})

def logout_view(request):
    """Log the user out and redirect to login page."""
    auth_logout(request)
    request.session.flush()
    return redirect('login_page')

# def phoenix_view(request):
#     return redirect('http://0.0.0.0:6006')

def phoenix_embed(request):
    if not _in_role(request, 'hr'):
        return HttpResponse('<div class="bidi-text">FORBIDDEN</div>', status=403)
    return render(request, 'phoenix_embed.html')

import json, queue, threading, time
from django.shortcuts import render
from django.http import JsonResponse
from langchain_huggingface import HuggingFaceEmbeddings
from utils.prepare_vectordb import PrepareVectorDB   # your new class
# global state exactly like the Gradio demo
STATE = {
    "loader": None,
    "raw_pages": None,
    "clean_pages": None,
    "chunks": None,
    "vectordb": None,
}
LOG_QUEUE = queue.Queue()

def log(msg, lvl="INFO"):
    LOG_QUEUE.put((lvl, msg))

# def trace_prepare(request):
#     """Render the trace page."""
#     if not request.session.get('user'):
#         return redirect('login_page')
#     return render(request, 'trace.html')

# def trace_api(request, step):
#     """Execute one step and stream logs."""
#     if not request.session.get('user'):
#         return JsonResponse({'error': 'login required'}, status=403)
#     def _run():
#         global STATE
#         try:
#             if step == "load":
#                 pdf_path = request.POST.get("pdf_path", "data/docs")
#                 embed_name = request.POST.get("embed_name", "sentence-transformers/all-MiniLM-L6-v2")
#                 chunk_size = int(request.POST.get("chunk_size", 1000))
#                 chunk_overlap = int(request.POST.get("chunk_overlap", 200))
#                 log("🚀 Instantiating PrepareVectorDB...")
#                 emb = HuggingFaceEmbeddings(model_name=embed_name)
#                 STATE["loader"] = PrepareVectorDB(
#                     data_directory=pdf_path,
#                     persist_directory="chroma_db",
#                     embedding_model_engine=emb,
#                     chunk_size=chunk_size,
#                     chunk_overlap=chunk_overlap,
#                 )
#                 log("✅ Instance ready – now loading pages...")
#                 STATE["raw_pages"] = STATE["loader"]._PrepareVectorDB__load_all_documents()
#                 log(f"📄 Loaded {len(STATE['raw_pages'])} raw pages")
#             elif step == "clean":
#                 if STATE["raw_pages"] is None:
#                     return log("No pages to clean – run Load first", "WARN")
#                 log("🔧 Cleaning + OCR...")
#                 STATE["clean_pages"] = STATE["loader"].process_pdf(STATE["raw_pages"])
#                 log(f"✅ Cleaning complete – {len(STATE['clean_pages'])} pages left")
#             elif step == "chunk":
#                 if STATE["clean_pages"] is None:
#                     return log("No cleaned pages – run Clean first", "WARN")
#                 log("📑 Chunking pages...")
#                 # your old _chunk_pages method – adapt if you renamed it
#                 STATE["chunks"] = STATE["loader"]._PrepareVectorDB__chunk_pages(STATE["clean_pages"])
#                 log(f"✅ Chunking complete – {len(STATE['chunks'])} chunks")
#             elif step == "store":
#                 if STATE["chunks"] is None:
#                     return log("No chunks – run Chunk first", "WARN")
#                 log("💾 Embedding & storing (this may take a while)...")
#                 STATE["vectordb"] = STATE["loader"].prepare_and_save_vectordb()
#                 log("🎉 Vector DB persisted to ./chroma_db")
#         except Exception as exc:
#             log(f"✖ � � {exc}", "ERROR")
#     # run step in background thread so we can stream logs
#     threading.Thread(target=_run, daemon=True).start()
#     # collect logs for 2 s (same trick as Gradio)
#     lines = []
#     stop = time.time() + 2
#     while time.time() < stop:
#         try:
#             lvl, msg = LOG_QUEUE.get(timeout=0.1)
#             color = {"INFO": "blue", "WARN": "orange", "ERROR": "red"}.get(lvl, "black")
#             lines.append(f'<span style="color:{color}">[{lvl}]</span> {msg}')
#         except queue.Empty:
#             pass
#     return JsonResponse({"console": "\n".join(lines), "step": step})

# ##########

def get_or_create_django_user(username, email, role):
    """Get or create a Django user for our hardcoded users"""
    user, created = User.objects.get_or_create(
        username=username,
        defaults={
            'email': email,
            'is_active': True,
            'is_staff': role == 'hr'
        }
    )
    if created:
        import secrets
        # Set random password for LDAP users (won't be used for auth)
        user.set_password(secrets.token_urlsafe(32))
        user.save()
    return user

def create_chat_session_in_db(user, title="New Chat", rag_mode="Preprocessed doc",
                               temperature=0.7, display_mode="answer_only"):
    """Create a new chat session in database"""
    session = ChatSession.objects.create(
        user=user,
        title=title,
        rag_mode=rag_mode,
        temperature=temperature,
        display_mode=display_mode
    )
    return session

def add_message_to_db(session, role, content, references=None, tokens_used=0):
    """Add a message to a session in database"""
    last_msg = ChatMessage.objects.filter(session=session).order_by('-sequence').first()
    sequence = last_msg.sequence + 1 if last_msg else 0
    message = ChatMessage.objects.create(
        session=session,
        role=role,
        content=content,
        sequence=sequence,
        references=references
    )
    # Update session metadata
    session.message_count += 1
    session.cumulative_tokens += tokens_used  # <-- CRITICAL UPDATE: Track cumulative tokens
    session.last_preview = content[:100] if content else ""
    session.updated_at = timezone.now()
    session.save()
    return message

def get_session_history_from_db(session, limit=20):
    """Get message history for a session from database"""
    messages = ChatMessage.objects.filter(session=session).order_by('sequence')[:limit]
    history = []
    temp_pair = []
    for msg in messages:
        if msg.role == 'user':
            if temp_pair:  # Save previous pair if exists
                history.append(temp_pair)
            temp_pair = []
            temp_pair.append(msg.content)
        elif msg.role == 'assistant' and temp_pair:
            temp_pair.append(msg.content)
            history.append(temp_pair)
            temp_pair = []
    if temp_pair:
        temp_pair.append("")
        history.append(temp_pair)
    return history

def init_user_session_storage(request):
    """Initialize session storage for chat sessions if not exists"""
    if 'chat_sessions' not in request.session:
        request.session['chat_sessions'] = {}
    if 'active_session' not in request.session:
        request.session['active_session'] = None

def create_new_session_in_storage(request, title="New Chat", rag_mode="Preprocessed doc",
                                   temperature=0.7, display_mode="answer_only"):
    """Create a new chat session in session storage"""
    session_id = str(uuid.uuid4())
    current_time = timezone.now().isoformat()
    request.session['chat_sessions'][session_id] = {
        'id': session_id,
        'title': title,
        'created_at': current_time,
        'last_activity': current_time,
        'history': [],
        'last_refs': '',
        'title_set': False,
        'rag_mode': rag_mode,
        'temperature': temperature,
        'display_mode': display_mode
    }
    request.session['active_session'] = session_id
    request.session.modified = True
    return session_id

def get_session_from_storage(request, session_id):
    """Get a session from session storage"""
    return request.session.get('chat_sessions', {}).get(session_id)

def update_session_in_storage(request, session_id, updates):
    """Update a session in storage"""
    if session_id in request.session.get('chat_sessions', {}):
        request.session['chat_sessions'][session_id].update(updates)
        request.session['chat_sessions'][session_id]['last_activity'] = timezone.now().isoformat()
        request.session.modified = True
        return True
    return False

def delete_session_from_storage(request, session_id):
    """Delete a session from storage"""
    if session_id in request.session.get('chat_sessions', {}):
        del request.session['chat_sessions'][session_id]
        if request.session.get('active_session') == session_id:
            remaining_sessions = list(request.session['chat_sessions'].keys())
            if remaining_sessions:
                request.session['active_session'] = remaining_sessions[0]
            else:
                request.session['active_session'] = None
        request.session.modified = True
        return True
    return False

def get_all_sessions_from_storage(request):
    """Get all sessions for the current user"""
    sessions_list = []
    chat_sessions = request.session.get('chat_sessions', {})
    active_session = request.session.get('active_session')
    for sid, session_data in chat_sessions.items():
        preview = 'Empty chat'
        history = session_data.get('history', [])
        if history:
            last_pair = history[-1] if history else []
            if last_pair and len(last_pair) > 0:
                last_message = last_pair[0] if last_pair[0] else (last_pair[1] if len(last_pair) > 1 else '')
                preview = (last_message[:50] + '...') if len(last_message) > 50 else last_message
        sessions_list.append({
            'id': sid,
            'title': session_data.get('title', 'Untitled'),
            'is_active': sid == active_session,
            'preview': preview,
            'created_at': session_data.get('created_at', ''),
            'message_count': len(history),
            'last_activity': session_data.get('last_activity', '')
        })
    return sessions_list

# ---------- Combined session functions (storage + database) ----------
def create_session_combined(request, title="New Chat", rag_mode="Preprocessed doc",
                             temperature=0.7, display_mode="answer_only", first_message=""):
    """Create session in both storage and database"""
    user = request.user if request.user.is_authenticated else None
    # Create in database if user is authenticated
    db_session = None
    if user:
        db_session = create_chat_session_in_db(
            user=user,
            title=title,
            rag_mode=rag_mode,
            temperature=temperature,
            display_mode=display_mode
        )
        session_id = str(db_session.id)
        # Add first message to database if provided
        if first_message:
            add_message_to_db(db_session, 'user', first_message)
    else:
        session_id = str(uuid.uuid4())
    # Create in session storage
    current_time = timezone.now().isoformat()
    request.session['chat_sessions'][session_id] = {
        'id': session_id,
        'title': title,
        'created_at': current_time,
        'last_activity': current_time,
        'history': [[first_message, '']] if first_message else [],
        'last_refs': '',
        'title_set': bool(first_message),
        'rag_mode': rag_mode,
        'temperature': temperature,
        'display_mode': display_mode,
        'db_session': bool(db_session)
    }
    request.session['active_session'] = session_id
    request.session.modified = True
    return session_id, db_session

def get_session_history_combined(session_id, request):
    """Get history from database if available, otherwise from storage"""
    user = request.user if request.user.is_authenticated else None
    # Try to get from database first
    if user:
        try:
            db_session = ChatSession.objects.get(id=session_id, user=user)
            return get_session_history_from_db(db_session)
        except (ChatSession.DoesNotExist, ValueError):
            pass
    # Fallback to session storage
    session_data = get_session_from_storage(request, session_id)
    return session_data.get('history', []) if session_data else []

def add_message_combined(session_id, request, role, content, references=None, tokens_used=0):
    """Add message to both database and session storage"""
    user = request.user if request.user.is_authenticated else None
    # Add to database if user is authenticated
    if user:
        try:
            db_session = ChatSession.objects.get(id=session_id, user=user)
            add_message_to_db(db_session, role, content, references, tokens_used)
        except (ChatSession.DoesNotExist, ValueError):
            pass
    # Add to session storage
    session_data = get_session_from_storage(request, session_id)
    if session_data:
        history = session_data.get('history', [])
        if role == 'user':
            history.append([content, ''])
        elif role == 'assistant' and history:
            history[-1][1] = content
        session_data['history'] = history
        if references:
            session_data['last_refs'] = references
        update_session_in_storage(request, session_id, session_data)

# ---------- Helper to reset user sessions ----------
def reset_user_sessions(user_id_or_username):
    """
    Delete all sessions for a specific user from both database and session storage.
    Args:
        user_id_or_username: Can be a username string or Django User object
    Returns:
        tuple: (success, message)
    """
    # If it's a username string, get the User object
    if isinstance(user_id_or_username, str):
        try:
            user = User.objects.get(username=user_id_or_username)
        except User.DoesNotExist:
            return False, f"User {user_id_or_username} not found"
    else:
        user = user_id_or_username
    # Delete all chat sessions and messages from database
    deleted_count = ChatSession.objects.filter(user=user).count()
    ChatSession.objects.filter(user=user).delete()
    # Note: ChatMessage objects will be deleted automatically due to CASCADE
    return True, f"Deleted {deleted_count} chat sessions for user {user.username}"

def determine_role_from_ldap(ldap_user):
    """
    Determine user role based on LDAP attributes.
    Customize this function based on your LDAP structure.
    """
    ldap_attrs = ldap_user.get('ldap_attrs', {})
    # Check department attribute
    if 'department' in ldap_attrs:
        dept = ldap_attrs['department'].lower()
        if 'hr' in dept or 'human resources' in dept or 'human_resources' in dept:
            return 'hr'
    # Check title attribute
    if 'title' in ldap_attrs:
        title = ldap_attrs['title'].lower()
        hr_titles = ['hr', 'human resources', 'recruiter', 'talent', '人事', 'موارد بشرية']
        if any(hr_title in title for hr_title in hr_titles):
            return 'hr'
    # Check memberOf for HR groups
    if 'memberOf' in ldap_attrs:
        member_of = ldap_attrs['memberOf']
        if isinstance(member_of, str):
            if any(term in member_of.lower() for term in ['hr', 'human_resources', 'hr_group']):
                return 'hr'
        elif isinstance(member_of, list):
            for group in member_of:
                if any(term in group.lower() for term in ['hr', 'human_resources', 'hr_group']):
                    return 'hr'
    # Default to employee role
    return 'emp'

def get_user_session_count(user, request):
    """Get total number of sessions for user (database + storage)"""
    count = 0
    # Count from database
    if user.is_authenticated:
        count += ChatSession.objects.filter(user=user).count()
    # Count from session storage
    storage_sessions = request.session.get('chat_sessions', {})
    # Only count storage sessions that aren't in database
    for session_id in storage_sessions:
        if user.is_authenticated:
            try:
                # Check if this exists in database
                ChatSession.objects.get(id=session_id, user=user)
                # If it exists in DB, don't count it (already counted)
                continue
            except (ChatSession.DoesNotExist, ValueError):
                pass
        count += 1
    return count

logger = logging.getLogger(__name__)
 
@require_http_methods(["POST"])
@csrf_exempt
def submit_rag_feedback(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    try:
        data = json.loads(request.body)
        print("?? Incoming request body:", json.dumps(data, indent=2))
        span_id = data.get('span_id')
        session_id = data.get('session_id')
        query = data.get('query', '')
        response = data.get('response', '')
        context_html = data.get('context_html', '')
        dimensions = data.get('dimensions', {})
        if not span_id:
            return JsonResponse({'error': 'span_id is required'}, status=400)
        user_id = request.user.username if request.user.is_authenticated else 'anonymous'
        # ????? ??? HTML
        context_chunks = []
        if context_html:
            import re
            clean_text = re.sub(r'<[^>]+>', '', context_html)
            if len(clean_text) > 1000:
                context_chunks = [clean_text[i:i+1000] for i in range(0, len(clean_text), 1000)]
            else:
                context_chunks = [clean_text] if clean_text else []
        dimension_names = {
            'context_relevance': 'Context Relevance',
            'context_sufficiency': 'Context Sufficiency',
            'faithfulness': 'Faithfulness',
            'answer_correctness': 'Answer Correctness',
            'answer_usefulness': 'Answer Usefulness',
            'comments':"comments"
        }
        score_map = {
            'Not relevant': 0.0,
            'Partially relevant': 0.5,
            'Clearly relevant': 1.0,
            'Not sufficient': 0.0,
            'Sufficient for a partial answer': 0.5,
            'Sufficient for a complete answer': 1.0,
            'Contains information not present in the context': 0.0,
            'Contains unstated inferences': 0.5,
            'Fully grounded in the context': 1.0,
            'Incorrect': 0.0,
            'Partially correct': 0.5,
            'Correct and accurate': 1.0,
            'Not useful': 0.0,
            'Partially useful': 0.5,
            'Useful and clear': 1.0,
        }
        results = []
        success_count = 0
        for dim_key, rating in dimensions.items():
            if not rating or dim_key not in dimension_names:
                continue
            score = score_map.get(rating, 0.5)
            dim_display = dimension_names[dim_key]
            annotation_core = {
                "span_id": span_id,
                "name": dim_key,
                "annotator_kind": "HUMAN",
                "result": {
                    "label": rating,
                    "score": score,
                    # "explanation": overall_feedback[:1000] if overall_feedback else None
                },
                "metadata": {
                    "session_id": session_id,
                    "user_id": user_id,
                    "feedback_source": "web_ui",
                    "dimension_display": dim_display,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "identifier": f"{session_id}_{dim_key}_{int(time.time())}"
            }
            annotation_payload = {
                "data": [annotation_core]
            }
            print(f"?? Sending {dim_key} payload:")
            print(json.dumps(annotation_payload, indent=2, default=str))
            try:
                url = "http://localhost:6006/v1/span_annotations?sync=false"
                resp = requests.post(
                    url,
                    json=annotation_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                print(f"?? Phoenix response ({dim_key}): {resp.status_code}")
                print("here : ",resp.text)
                if resp.status_code == 422:
                    print("? Validation error from Phoenix")
                    results.append({
                        "dimension": dim_key,
                        "status": "failed",
                        "error": resp.text[:300]
                    })
                else:
                    resp.raise_for_status()
                    success_count += 1
                    results.append({
                        "dimension": dim_key,
                        "status": "success"
                    })
            except Exception as e:
                print(f"?? Exception while sending {dim_key}: {str(e)}")
                results.append({
                    "dimension": dim_key,
                    "status": "error",
                    "error": str(e)
                })
        return JsonResponse({
            'success': success_count > 0,
            'message': f'?? ????? {success_count} ?? ??? {len([d for d in dimensions.values() if d])} ?????',
            'details': results
        })
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        print("?? Fatal error in submit_rag_feedback:", str(e))
        return JsonResponse({'error': str(e)}, status=500)