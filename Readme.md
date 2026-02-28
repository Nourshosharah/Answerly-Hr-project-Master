# HR System with RAG Chatbot

A Django-based HR system featuring LDAP authentication and a Retrieval-Augmented Generation (RAG) chatbot for answering employee queries based on HR documentation. The system provides role-based access control for HR administrators and employees, with comprehensive evaluation capabilities using TruLens and Phoenix tracing.

## Overview

This project is a corporate HR support system that enables employees to get instant answers to their HR-related questions through an intelligent chatbot interface. The chatbot leverages RAG technology to provide accurate responses based on company HR documents, policies, and procedures. The system supports LDAP integration for enterprise authentication and includes advanced evaluation and monitoring features.

## Features

* **Intelligent Chatbot:** Employees can ask questions about HR policies, procedures, and documentation, receiving accurate, context-aware responses generated from the company’s knowledge base.
* **LDAP Authentication:** Integration with Active Directory allows users to log in with corporate credentials.
* **Role-Based Access Control:** Distinguishes HR administrators (full access to document management, analytics, and session control) from regular employees (chatbot access only).
* **Persistent Sessions:** Chat history is stored in both database and session storage, allowing conversations to resume across logins.
* **Evaluation & Monitoring:** TruLens evaluates RAG responses for relevance, correctness, and faithfulness; Phoenix provides detailed tracing and observability.
* **Resource Tracking:** Tracks token usage and estimated costs per conversation.

## Technology Stack

* **Backend:** Django 5.2.8
* **Authentication:** LDAP/Active Directory via `django-auth-ldap`
* **Database:** PostgreSQL 18.3 (production), SQLite (development)
* **RAG & AI Components:**

  * LangChain for RAG pipeline
  * Chroma for vector database
  * HuggingFace multilingual E5-large embeddings
  * Generation model: GPT-20B OSS
* **Evaluation & Tracing:** TruLens and Phoenix
* **Frontend:** Vanilla JS + Tailwind CSS
* **Python:** 3.12+

## Project Structure

```
hr_system/      # Django project settings, URLs, WSGI/ASGI
ragsys/         # Main Django app: models, views, auth, URLs, templates, static files
utils/          # Utilities: chatbot logic, vector DB prep, config loader, evaluation, data upload
```

Environment-specific configuration (LDAP credentials, API keys) is managed via `.env` files and `utils/app_config.yml`.

## Installation

1. **Clone the repository** and create a virtual environment:

```bash
git clone <repo-url>
cd hr-system
```

3. **System-level packages (Linux/Ubuntu/Debian):**

```bash
sudo apt-get install libpq-dev libxml2-dev libxslt1-dev libmagic1 poppler-utils tesseract-ocr
```

## Configuration

### Environment Variables (`.env`)

```env
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,ip_addresses

# LDAP
LDAP_SERVER=ldap://your-ldap-server
LDAP_PORT=389
LDAP_SEARCH_BASE=DC=yourcompany,DC=local
LDAP_BIND_DN=CN=service-account,CN=Users,DC=yourcompany,DC=local
LDAP_BIND_PASSWORD=your-service-account-password

# vLLM
VLLM_BASE_URL=http://localhost:8001

# Phoenix
PHOENIX_ENDPOINT=http://localhost:6006
```

### Application Configuration (`utils/app_config.yml`)

```yaml
data_directory: "utils/data/docs"
persist_directory: "chroma_db"
embedding_model_name: "intfloat/multilingual-e5-large"
chunk_size: 1000
chunk_overlap: 200
generation_model_name: "gpt-20b-oss"
```

### Django Settings (`hr_system/settings.py`)

* `DATABASES`: Configure PostgreSQL 18.3 for production
* `LDAP_*`: LDAP server credentials
* `ALLOWED_HOSTS` & `DEBUG` as per environment

## Database Setup

```bash
python manage.py migrate
python manage.py createsuperuser
```

## Running the Application

### Start Django Server

```bash
python manage.py runserver 0.0.0.0:8000
```

### vLLM Server for GPT-20B OSS Model

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model ./my-20b-model \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager
```

### Required Services

1. Django Web Application
2. vLLM Server (GPT-20B OSS)
3. Phoenix Server (tracing & observability)

## Usage

* **Login:** Via LDAP credentials; roles determined by LDAP attributes.
* **Employees:** Ask HR-related questions using the RAG chatbot; conversations stored persistently.
* **HR Administrators:** Upload/manage HR documents, monitor usage, access evaluation metrics, and view Phoenix traces.

## API Endpoints

| Endpoint                   | Method | Description                     |
| -------------------------- | ------ | ------------------------------- |
| `/api/chat/`               | POST   | Send a message to the chatbot   |
| `/api/chat/sessions/`      | GET    | Get all chat sessions           |
| `/api/chat/sessions/`      | POST   | Create a new chat session       |
| `/api/chat/sessions/<id>/` | DELETE | Delete a chat session           |
| `/api/evaluation/metrics/` | GET    | Retrieve evaluation metrics     |
| `/api/evaluation/submit/`  | POST   | Submit user feedback            |
| `/api/data/trigger/`       | POST   | Trigger data indexing (HR only) |
| `/api/data/status/`        | GET    | Get data indexing status        |

## Evaluation & Monitoring

* **TruLens:** Automatic evaluation of RAG responses for relevance, faithfulness, correctness, and usefulness.
* **Phoenix:** Distributed tracing with spans and annotations for performance monitoring.

## Development

* **Tests:** `python manage.py test`
* **Type Checking:** `mypy .`

## License

Internal corporate use only. All rights reserved.

## Support

Contact IT or the HR system administrator for assistance.

## Acknowledgments

This project uses: Django, LangChain, Chroma, TruLens, Phoenix, HuggingFace Transformers.
