# **Content-based Recommendation Systems**
BẠN CÓ THỂ XEM TÀI LIỆU BẰNG:


1.   Github  &nbsp; <a href="" role="button"><img class="notebook-badge-image" src="https://img.shields.io/static/v1?label=&amp;message=View%20On%20GitHub&amp;color=586069&amp;logo=github&amp;labelColor=2f363d"></a>&nbsp;
2.   Colab &nbsp; <a href="https://colab.research.google.com/drive/1DU7lAB_jtASTAS5V6aEqDfQ0VaW1Kg6L?usp=sharing"><img class="notebook-badge-image" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## **INTRODUCTION**
 Mùa thu này thời tiết rất dễ chịu, mình sẽ hướng dẫn cách để trở thành một ông thầy "bói toán" nghiệp dư haha. Bói toán, giải mộng đó là cách dự đoán một sự kiện nào đó ở tương lai, quá khứ mà chúng ta khó nắm được, sờ được :)). Những người thầy bói toán giỏi nhất, nổi tiếng nhất trên thế giới là ai? Bà Ba béo bán bánh bèo ở gốc đa làng bạn hay bà tiên tri Baba Vanga thần thánh ...? Nhưng có những ông thầy mà ai cũng biết đó là Facebook, Google, Netflix, Youtube...ông thầy này rất giỏi, biết hầu hết mọi thứ cua bạn, đôi khi còn biết số lượng bồ nhí khủng bố xinh đẹp của bạn, còn dự đoán được hành động.... nói chung là mọi thư của bạn :)))). Rốt cuộc là để hành nghề bói toán trong cuộc sống bon chen này bạn cần có một tâm hồn đẹp và khả năng dự đoán vô cùng chính xác.
 ## **Mô tả bài toán**
 Hôm nay là ngày hành nghề đầu tiên của bạn trên giang hồ "bói toán", bạn ngồi dật dẹo dưới gốc đa làng cạnh giếng đình chờ con mồi tới.. à khách tới, ngồi tê cả mông mà chả có ma nào hỏi han gì. Cuối ngày đang định ra về thì có một anh web phim xxx lò dò tới bên cạnh nói:
 "Tôi muốn "bói" cho số lượng lớn người dùng xem phim trên web của tôi, tôi muốn biết họ có thể thích hay không thích videos nào, tôi có bộ cơ sở dữ liệu MovieLens 100k đánh giá của người dùng. Anh dựa vào dữ liệu để bói cho tất cả người dùng, nếu làm tốt tôi sẽ làm cho cuộc đời anh nở hoa"
 Bạn mừng rơi H20 mắt vội yes yes... liên tục và cong mông chạy về nhà mở jupyter notebook lên bắt đầu code... :)))

<a href="https://grouplens.org/datasets/movielens/100k/">Bộ cơ sở dữ liệu MovieLens 100k</a>

Sau khi download và giải nén, chúng ta sẽ thu được rất nhiều các file nhỏ, chúng ta chỉ cần quan tâm các file sau:

* u.data: Chứa toàn bộ các ratings của 943 users cho 1682 movies. Mỗi user rate ít nhất 20 movies. Thông tin về thời gian rate cũng được cho nhưng chúng ta không sử dụng trong bài viết này.

* ua.base, ua.test, ub.base, ub.test: là hai cách chia toàn bộ dữ liệu ra thành hai tập con, một cho training, một cho test. Chúng ta sẽ thực hành trên ua.base và ua.test. Bạn đọc có thể thử với cách chia dữ liệu còn lại.

* u.user: Chứa thông tin về users, bao gồm: id, tuổi, giới tính, nghề nghiệp, zipcode (vùng miền), vì những thông tin này cũng có thể ảnh hưởng tới sở thích của các users. Tuy nhiên, trong bài viết này, chúng ta sẽ không sử dụng các thông tin này, trừ thông tin về id để xác định các user khác nhau.

* u.genre: Chứa tên của 19 thể loại phim. Các thể loại bao gồm: unknown, Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western,

* u.item: thông tin về mỗi bộ phim

##**Bắt đầu xây dựng "quẻ" bói :))**
Với cơ sở dữ liệu này, chúng ta sẽ sử dụng thư viện pandas để trích xuất dữ liệu, có thể được cài đặt bằng pip install pandas.





 



```python
import pandas as pd 
import numpy as np
#Reading user file:
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

n_users = users.shape[0]
print ('Number of users:', n_users)
# users.head() #uncomment this to see some few examples
```


```python
#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base
rate_test = ratings_test

print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])
```


```python
#Công việc quan trọng trong content-based recommendation system là xây dựng profile cho mỗi item,tức feature vector cho mỗi item.
#Trước hết, chúng ta cần load toàn bộ thông tin về các items vào biến items:
#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

n_items = items.shape[0]
print('Number of items:', n_items)
```


```python
#Vì ta đang dựa trên thể loại của phim để xây dựng profile, ta sẽ chỉ quan tâm tới 19 giá trị nhị phân ở cuối mỗi hàng:
X0 = items
X_train_counts = X0.iloc[:, -19:]
```


```python
#Tiếp theo, chúng ta sẽ xây dựng feature vector cho mỗi item dựa trên ma trận thể loại phim và feature TF-IDF.
#Tôi sẽ mô tả kỹ hơn về TF-IDF trong một bài viết khác. Tạm thời, chúng ta sử dụng thư viện sklearn.
#tfidf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.values).toarray()
```


```python
'''
Sau bước này, mỗi hàng của tfidf tương ứng với feature vector của một bộ phim.

Tiếp theo, với mỗi user, chúng ta cần đi tìm những bộ phim nào mà user đó đã rated, và giá trị của các rating đó.

'''
import numpy as np
def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix.iloc[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    ids = np.where(y == user_id +1)[0] 
    item_ids = rate_matrix.iloc[ids, 1] - 1 # index starts from 0 
    scores = rate_matrix.iloc[ids, 2]
    return (item_ids, scores)
```


```python
#Bây giờ, ta có thể đi tìm các hệ số của Ridge Regression cho mỗi user:

from sklearn.linear_model import Ridge
from sklearn import linear_model

d = tfidf.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

for n in range(n_users):    
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept  = True)
    Xhat = tfidf[ids, :]
    
    clf.fit(Xhat, scores) 
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_

```


```python
#Sau khi tính được các hệ số W và b, ratings cho mỗi items được dự đoán bằng cách tính:
# predicted scores
Yhat = tfidf.dot(W) + b
```


```python
#Dưới đây là một ví dụ với user có id là 10.

n = 10
np.set_printoptions(precision=2) # 2 digits after . 
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]
print('Rated movies ids :', ids )
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[ids, n])
```


```python
#Để đánh giá mô hình tìm được, chúng ta sẽ sử dụng Root Mean Squared Error (RMSE),
#tức căn bậc hai của trung bình cộng bình phương của lỗi. Lỗi được tính là hiệu của true rating và predicted rating:
#from past.builtins import xrange
import math
def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred 
        se += (e*e).sum(axis = 0)
        cnt += e.size 
    return math.sqrt(se/cnt)

print('RMSE for training:', evaluate(Yhat, rate_train, W, b))
print('RMSE for test:', evaluate(Yhat, rate_test, W, b))
```

Cảm ơn mọi người đã đọc bài!
Tài liệu bài viết này mình tham khảo từ: https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/
