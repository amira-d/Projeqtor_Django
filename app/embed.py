class EmbedToken:
    def __init__(self, report_id, group_id, settings=None):
        self.username = 'amira.doghri@esprit-tn.com'
        self.password = 'Xar65632'
        self.client_id = '219b6c97-3c14-4b5d-aa2d-612fb0aa4c9f'
        self.report_id = 'a3ec38c1-522f-4a49-8c76-26fb0caba298'
        self.group_id = '25bedcd7-8366-4c6d-8ce7-2a2bb4b48987'
        self.client_secret = 'fA6s3Q~o53t8ozQKA_XURh_lBlk35d_P9X'
        if settings is None:
            self.settings = {'accessLevel': 'View', 'allowSaveAs': 'false'}
        else:
            self.settings = settings
        self.access_token = self.get_access_token()
        self.config = self.get_embed_token()

    def get_access_token(self):
        data = {
            'grant_type': 'password',
            'scope': 'https://graph.microsoft.com/.default',
            'resource': r'https://analysis.windows.net/powerbi/api',
            'client_id': self.client_id,
            'username': self.username,
            'password': self.password,
            'client_secret' : self.client_secret
        }
        response = requests.post('https://login.microsoftonline.com/common/oauth2/token', data=data)
        return response.json().get('access_token')

    def get_embed_token(self):
        dest = 'https://api.powerbi.com/v1.0/myorg/groups/' + self.group_id \
               + '/reports/' + self.report_id + '/GenerateToken'
        embed_url = 'https://app.powerbi.com/reportEmbed?reportId=' \
                    + self.report_id + '&groupId=' + self.group_id
        headers = {'Authorization': 'Bearer ' + self.access_token}
        response = requests.post(dest, data=self.settings, headers=headers)
        self.token = response.json().get('token')
        return {'token': self.token, 'embed_url': embed_url, 'report_id': self.report_id}

    def get_report(self):
        headers = {'Authorization': 'Bearer ' + self.token}
        dest = 'https://app.powerbi.com/reportEmbed?reportId=' + self.report_id + \
               '&amp;groupId=' + self.group_id
        response = requests.get(dest, data=self.settings, headers=headers)
        return response