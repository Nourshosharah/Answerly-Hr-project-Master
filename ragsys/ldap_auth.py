# ragsys/ldap_auth.py

import ldap3
from ldap3.core.exceptions import LDAPException
from django.conf import settings
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)


class LDAPAuth:
    """Active Directory authentication using ldap3"""

    def __init__(self):
        self.server_uri = settings.LDAP_SERVER
        self.port = settings.LDAP_PORT
        self.use_ssl = settings.LDAP_USE_SSL
        self.search_base = settings.LDAP_SEARCH_BASE
        self.search_filter = settings.LDAP_SEARCH_FILTER
        self.attributes = settings.LDAP_ATTRIBUTES
        self.bind_dn = settings.LDAP_BIND_DN
        self.bind_password = settings.LDAP_BIND_PASSWORD
        self.user_mapping = settings.LDAP_USER_MAPPING

    def get_server(self):
        return ldap3.Server(
            self.server_uri,
            port=self.port,
            use_ssl=self.use_ssl
        )

    def authenticate(self, username, password):
        if not username or not password:
            return None

        cache_key = f'ldap_user_{username}'
        cached = cache.get(cache_key)
        if cached:
            return cached

        try:
            server = self.get_server()

            # 1️⃣ Bind using service account
            conn = ldap3.Connection(
                server,
                user=self.bind_dn,
                password=self.bind_password,
                authentication=ldap3.SIMPLE,
                auto_bind=True,
            )

            # 2️⃣ Search for user
            search_filter = self.search_filter.format(username)

            conn.search(
                search_base=self.search_base,
                search_filter=search_filter,
                attributes=self.attributes,
                size_limit=1,
            )

            if not conn.entries:
                logger.warning(f"LDAP user not found: {username}")
                conn.unbind()
                return None

            entry = conn.entries[0]
            user_dn = entry.entry_dn

            ldap_attrs = {}
            for attr in self.attributes:
                if hasattr(entry, attr):
                    value = getattr(entry, attr)
                    if value:
                        ldap_attrs[attr] = str(value)

            conn.unbind()

            # 3️⃣ Bind as user (PASSWORD CHECK)
            user_conn = ldap3.Connection(
                server,
                user=user_dn,
                password=password,
                authentication=ldap3.SIMPLE,
            )

            if not user_conn.bind():
                logger.warning(f"Invalid LDAP credentials for {username}")
                user_conn.unbind()
                return None

            user_conn.unbind()

            # 4️⃣ Map LDAP → app user
            user_info = self.map_ldap_to_user(ldap_attrs, username)

            cache.set(
                cache_key,
                user_info,
                settings.LDAP_CACHE_TIMEOUT
            )

            return user_info

        except LDAPException as e:
            logger.error(f"LDAP error for {username}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected LDAP auth error for {username}")
            return None

    def map_ldap_to_user(self, ldap_attrs, username):
        user = {
            'username': username,
            'email': '',
            'first_name': '',
            'last_name': '',
            'ldap_attrs': ldap_attrs,
        }

        for django_field, ldap_field in self.user_mapping.items():
            if ldap_field in ldap_attrs:
                user[django_field] = ldap_attrs[ldap_field]

        if not user['email']:
            user['email'] = f'{username}@itlab.local'

        return user
