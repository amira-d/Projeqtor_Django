def index(request):
    conf = EmbedToken(report.get('report_id'), report.get('group_id'))
    token = conf.get_embed_token()
    return render(request, 'unfi_breakout/unfi_breakout.html', {'selectedReport': token.get('report_id'),
                                                                'embedToken': token.get('token')})