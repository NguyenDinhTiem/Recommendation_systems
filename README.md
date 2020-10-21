{
      "cell_type": "markdown",
      "metadata": {
        "id": "Jxnuu80q_EPP"
      },
      "source": [
        "# **Content-based Recommendation Systems**\n",
        "BẠN CÓ THỂ XEM TÀI LIỆU BẰNG:\n",
        "\n",
        "\n",
        "1.   Github  &nbsp; <a href=\"\" role=\"button\"><img class=\"notebook-badge-image\" src=\"https://img.shields.io/static/v1?label=&amp;message=View%20On%20GitHub&amp;color=586069&amp;logo=github&amp;labelColor=2f363d\"></a>&nbsp;\n",
        "2.   Colab &nbsp; <a href=\"https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/\"><img class=\"notebook-badge-image\" src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"></a>\n",
        "\n",
        "## **INTRODUCTION**\n",
        " Mùa thu này thời tiết rất dễ chịu, mình sẽ hướng dẫn cách để trở thành một ông thầy \"bói toán\" nghiệp dư haha. Bói toán, giải mộng đó là cách dự đoán một sự kiện nào đó ở tương lai, quá khứ mà chúng ta khó nắm được, sờ được :)). Những người thầy bói toán giỏi nhất, nổi tiếng nhất trên thế giới là ai? Bà Ba béo bán bánh bèo ở gốc đa làng bạn hay bà tiên tri Baba Vanga thần thánh ...? Nhưng có những ông thầy mà ai cũng biết đó là Facebook, Google, Netflix, Youtube...ông thầy này rất giỏi, biết hầu hết mọi thứ cua bạn, đôi khi còn biết số lượng bồ nhí khủng bố xinh đẹp của bạn, còn dự đoán được hành động.... nói chung là mọi thư của bạn :)))). Rốt cuộc là để hành nghề bói toán trong cuộc sống bon chen này bạn cần có một tâm hồn đẹp và khả năng dự đoán vô cùng chính xác.\n",
        " ## **Mô tả bài toán**\n",
        " Hôm nay là ngày hành nghề đầu tiên của bạn trên giang hồ \"bói toán\", bạn ngồi dật dẹo dưới gốc đa làng cạnh giếng đình chờ con mồi tới.. à khách tới, ngồi tê cả mông mà chả có ma nào hỏi han gì. Cuối ngày đang định ra về thì có một anh web phim xxx lò dò tới bên cạnh nói:\n",
        " \"Tôi muốn \"bói\" cho số lượng lớn người dùng xem phim trên web của tôi, tôi muốn biết họ có thể thích hay không thích videos nào, tôi có bộ cơ sở dữ liệu MovieLens 100k đánh giá của người dùng. Anh dựa vào dữ liệu để bói cho tất cả người dùng, nếu làm tốt tôi sẽ làm cho cuộc đời anh nở hoa\"\n",
        " Bạn mừng rơi H20 mắt vội yes yes... liên tục và cong mông chạy về nhà mở jupyter notebook lên bắt đầu code... :)))\n",
        "\n",
        "<a href=\"https://grouplens.org/datasets/movielens/100k/\">Bộ cơ sở dữ liệu MovieLens 100k</a>\n",
        "\n",
        "Sau khi download và giải nén, chúng ta sẽ thu được rất nhiều các file nhỏ, chúng ta chỉ cần quan tâm các file sau:\n",
        "\n",
        "* u.data: Chứa toàn bộ các ratings của 943 users cho 1682 movies. Mỗi user rate ít nhất 20 movies. Thông tin về thời gian rate cũng được cho nhưng chúng ta không sử dụng trong bài viết này.\n",
        "\n",
        "* ua.base, ua.test, ub.base, ub.test: là hai cách chia toàn bộ dữ liệu ra thành hai tập con, một cho training, một cho test. Chúng ta sẽ thực hành trên ua.base và ua.test. Bạn đọc có thể thử với cách chia dữ liệu còn lại.\n",
        "\n",
        "* u.user: Chứa thông tin về users, bao gồm: id, tuổi, giới tính, nghề nghiệp, zipcode (vùng miền), vì những thông tin này cũng có thể ảnh hưởng tới sở thích của các users. Tuy nhiên, trong bài viết này, chúng ta sẽ không sử dụng các thông tin này, trừ thông tin về id để xác định các user khác nhau.\n",
        "\n",
        "* u.genre: Chứa tên của 19 thể loại phim. Các thể loại bao gồm: unknown, Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western,\n",
        "\n",
        "* u.item: thông tin về mỗi bộ phim\n",
        "\n",
        "##**Bắt đầu xây dựng \"quẻ\" bói :))**\n",
        "Với cơ sở dữ liệu này, chúng ta sẽ sử dụng thư viện pandas để trích xuất dữ liệu, có thể được cài đặt bằng pip install pandas.\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        " \n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xa0c0E0QQVXK"
      },
      "source": [
        "import pandas as pd \n",
        "import numpy as np\n",
        "#Reading user file:\n",
        "u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']\n",
        "users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,\n",
        " encoding='latin-1')\n",
        "\n",
        "n_users = users.shape[0]\n",
        "print ('Number of users:', n_users)\n",
        "# users.head() #uncomment this to see some few examples"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GRvmscnPQX93"
      },
      "source": [
        "#Reading ratings file:\n",
        "r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']\n",
        "\n",
        "ratings_base = pd.read_csv('ml-100k/ua.base', sep='\\t', names=r_cols, encoding='latin-1')\n",
        "ratings_test = pd.read_csv('ml-100k/ua.test', sep='\\t', names=r_cols, encoding='latin-1')\n",
        "\n",
        "rate_train = ratings_base\n",
        "rate_test = ratings_test\n",
        "\n",
        "print('Number of traing rates:', rate_train.shape[0])\n",
        "print('Number of test rates:', rate_test.shape[0])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ib_XGYRTQhbL"
      },
      "source": [
        "#Công việc quan trọng trong content-based recommendation system là xây dựng profile cho mỗi item,tức feature vector cho mỗi item.\n",
        "#Trước hết, chúng ta cần load toàn bộ thông tin về các items vào biến items:\n",
        "#Reading items file:\n",
        "i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',\n",
        " 'Animation', 'Children\\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',\n",
        " 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']\n",
        "\n",
        "items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')\n",
        "\n",
        "n_items = items.shape[0]\n",
        "print('Number of items:', n_items)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7IL792LVQidD"
      },
      "source": [
        "#Vì ta đang dựa trên thể loại của phim để xây dựng profile, ta sẽ chỉ quan tâm tới 19 giá trị nhị phân ở cuối mỗi hàng:\n",
        "X0 = items\n",
        "X_train_counts = X0.iloc[:, -19:]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "L0auEzRjQoN4"
      },
      "source": [
        "#Tiếp theo, chúng ta sẽ xây dựng feature vector cho mỗi item dựa trên ma trận thể loại phim và feature TF-IDF.\n",
        "#Tôi sẽ mô tả kỹ hơn về TF-IDF trong một bài viết khác. Tạm thời, chúng ta sử dụng thư viện sklearn.\n",
        "#tfidf\n",
        "from sklearn.feature_extraction.text import TfidfTransformer\n",
        "transformer = TfidfTransformer(smooth_idf=True, norm ='l2')\n",
        "tfidf = transformer.fit_transform(X_train_counts.values).toarray()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AStALAi1Qsf3"
      },
      "source": [
        "'''\n",
        "Sau bước này, mỗi hàng của tfidf tương ứng với feature vector của một bộ phim.\n",
        "\n",
        "Tiếp theo, với mỗi user, chúng ta cần đi tìm những bộ phim nào mà user đó đã rated, và giá trị của các rating đó.\n",
        "\n",
        "'''\n",
        "import numpy as np\n",
        "def get_items_rated_by_user(rate_matrix, user_id):\n",
        "    \"\"\"\n",
        "    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp\n",
        "    we care about the first three values\n",
        "    return (item_ids, scores) rated by user user_id\n",
        "    \"\"\"\n",
        "    y = rate_matrix.iloc[:,0] # all users\n",
        "    # item indices rated by user_id\n",
        "    # we need to +1 to user_id since in the rate_matrix, id starts from 1 \n",
        "    # while index in python starts from 0\n",
        "    ids = np.where(y == user_id +1)[0] \n",
        "    item_ids = rate_matrix.iloc[ids, 1] - 1 # index starts from 0 \n",
        "    scores = rate_matrix.iloc[ids, 2]\n",
        "    return (item_ids, scores)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zTtjstX4Qve4"
      },
      "source": [
        "#Bây giờ, ta có thể đi tìm các hệ số của Ridge Regression cho mỗi user:\n",
        "\n",
        "from sklearn.linear_model import Ridge\n",
        "from sklearn import linear_model\n",
        "\n",
        "d = tfidf.shape[1] # data dimension\n",
        "W = np.zeros((d, n_users))\n",
        "b = np.zeros((1, n_users))\n",
        "\n",
        "for n in range(n_users):    \n",
        "    ids, scores = get_items_rated_by_user(rate_train, n)\n",
        "    clf = Ridge(alpha=0.01, fit_intercept  = True)\n",
        "    Xhat = tfidf[ids, :]\n",
        "    \n",
        "    clf.fit(Xhat, scores) \n",
        "    W[:, n] = clf.coef_\n",
        "    b[0, n] = clf.intercept_\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "E8tReNbbQySw"
      },
      "source": [
        "#Sau khi tính được các hệ số W và b, ratings cho mỗi items được dự đoán bằng cách tính:\n",
        "# predicted scores\n",
        "Yhat = tfidf.dot(W) + b"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FwKGMk-2RAoK"
      },
      "source": [
        "#Dưới đây là một ví dụ với user có id là 10.\n",
        "\n",
        "n = 10\n",
        "np.set_printoptions(precision=2) # 2 digits after . \n",
        "ids, scores = get_items_rated_by_user(rate_test, n)\n",
        "Yhat[n, ids]\n",
        "print('Rated movies ids :', ids )\n",
        "print('True ratings     :', scores)\n",
        "print('Predicted ratings:', Yhat[ids, n])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vIh2ujnBQ6Fl"
      },
      "source": [
        "#Để đánh giá mô hình tìm được, chúng ta sẽ sử dụng Root Mean Squared Error (RMSE),\n",
        "#tức căn bậc hai của trung bình cộng bình phương của lỗi. Lỗi được tính là hiệu của true rating và predicted rating:\n",
        "#from past.builtins import xrange\n",
        "import math\n",
        "def evaluate(Yhat, rates, W, b):\n",
        "    se = 0\n",
        "    cnt = 0\n",
        "    for n in range(n_users):\n",
        "        ids, scores_truth = get_items_rated_by_user(rates, n)\n",
        "        scores_pred = Yhat[ids, n]\n",
        "        e = scores_truth - scores_pred \n",
        "        se += (e*e).sum(axis = 0)\n",
        "        cnt += e.size \n",
        "    return math.sqrt(se/cnt)\n",
        "\n",
        "print('RMSE for training:', evaluate(Yhat, rate_train, W, b))\n",
        "print('RMSE for test:', evaluate(Yhat, rate_test, W, b))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "T21rQ8mbR-an"
      },
      "source": [
        "Cảm ơn mọi người đã đọc bài!\n",
        "Tài liệu bài viết này mình tham khảo từ: https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/"
      ]
    }
