from SearchEngine import SearchEng
from django.shortcuts import render
from django.views import generic
from django import forms


class SubmitForm(forms.Form):
    err_msg = forms.CharField(label='错误信息')
    scene = forms.CharField(label='场景', required=False)


class SearchView(generic.View):
    form_class = SubmitForm
    template_name = 'index.html'

    def get(self, request):
        return render(request, self.template_name, {'has_cluster': False, 'has_article': False, 'err_msg_placeholder': "", 'scene_placeholder': ""})

    def post(self, request):
        form_data2 = self.form_class(request.POST)
        if form_data2.is_valid():
            render_data = dict()
            # 查找输入的错误信息、场景信息对应的解决方法及其文章出处
            form_data = form_data2.cleaned_data
            err_msg = form_data["err_msg"]
            scene = form_data["scene"]
            render_data['err_msg_placeholder'] = err_msg
            render_data['scene_placeholder'] = scene
            aid_list, solve_msgs = SearchEng.get_all_solves(err_msg, scene)
            ranked_common_sstr_origin, ranked_aid_list = SearchEng.solve_statistic(aid_list, solve_msgs)
            # 构造解决方案的输出内容
            if ranked_common_sstr_origin:
                render_data['has_cluster'] = True
                render_data['cluster'] = []
                for t in range(len(ranked_common_sstr_origin)):
                    common_sstr_dic = {'text': ranked_common_sstr_origin[t], 'num': len(ranked_aid_list[t]), 'article': []}
                    for aid in ranked_aid_list[t]:
                        common_sstr_dic['article'].append({'title': SearchEng.base_data[aid].title, 'link': SearchEng.base_data[aid].link})
                    render_data['cluster'].append(common_sstr_dic)
            else:
                render_data['has_cluster'] = False
            # 构造全部解决方案的链接
            if aid_list:
                render_data['has_article'] = True
                render_data['all_articles'] = []
                if len(aid_list) > 100:
                    article_num = 100
                    render_data['article_num_msg'] = "前100"
                else:
                    article_num = len(aid_list)
                    render_data['article_num_msg'] = "全部" + str(article_num)
                for t in range(article_num):
                    aid = aid_list[t]
                    solve_dic = {'title': SearchEng.base_data[aid].title, 'link': SearchEng.base_data[aid].link}
                    solve_msg_1article = solve_msgs[t].replace('\n', ' ')
                    if len(solve_msg_1article) >= 100:
                        solve_dic['solve_msg'] = solve_msg_1article[:100]
                    else:
                        solve_dic['solve_msg'] = solve_msg_1article
                    render_data['all_articles'].append(solve_dic)
            else:
                render_data['has_article'] = False
            return render(request, self.template_name, render_data)
        else:
            pass
