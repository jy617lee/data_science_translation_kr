# Learning Market Dynamics for Optimal Pricing

## Combining elements of machine learning with structural modeling

![img](https://cdn-images-1.medium.com/max/2000/1*9qPQ_U7hyqaAPEJBvkZOiw.png)

How do you think market dynamics influences this beautiful home in Cape Town?

market dynamics가 케이프타운에 있는 이 아름다운 집에 어떻게 영향을 미칠까?

Market dynamics plays a key role in matching guests with hosts in two-sided marketplaces such as Airbnb. 

Market dynamics는 에어비앤비와 같은 two-sided marketplaces에서 게스트와 호스트를 matching시키는데 중요한 역할을 한다. 

Supply and demand vary drastically across different locations, different check in dates and different lead times until check-in. 

위치, 체크인 날짜, 체크인까지 걸리는 lead time에 따라 공급과 수요가 크게 달라진다 

It is important for us to understand and forecast these spatial and temporal trends in order to find better matches for our community of hosts and guests.

에어비앤비 커뮤니티의 호스트와 게스트를 더 잘 매칭해주기 위해서는 이런 공간적, 시간적 트렌드를 이해하고 예측하는 것이 중요하다. 

In this post, we describe a framework used to model lead time dynamics in order to help hosts price their homes more competitively and improve their earnings potential.

이 글에서는 호스트가 가격을 경쟁력있게 책정하고 잠재 수익을 향상시킬 수 있도록 하는 lead time dynamics를 모델링하기 위한 프레임워크에 대해 설명한다.

 We embrace both machine learning (ML) and structural modeling to achieve improved predictive performance and model interpretability.

우리는 예측 성능과 모델의 해석 가능성을 높이기 위해 기계학습(ML)과 구조 모델링을 채택한다. 



### A Primer on Lead Time Dynamics

![img](https://cdn-images-1.medium.com/max/900/1*oGd2Q_gx8UNHnKUpbyp7wg.png)

The lead time for a booking refers to the time between the date of booking and the trip check-in date. Taking a trip with a check-in date of New Year’s Eve (December 31) as an example, when a guest books 30 days in advance (on December 1), the booking lead time is 30 days. On the other hand, the booking lead time would be 0 for guests who make a last minute reservation on the day of check-in.

Guests will continue to make bookings for the New Year’s Eve as time progresses and the booking date gets closer and closer to the end of the year. This booking process reflects the inflow of demand and can be treated as a stochastic **arrival process**. The corresponding distribution of bookings over lead time is the booking **lead time distribution**.

#### Why Model the Lead Time Distribution?

![img](https://cdn-images-1.medium.com/max/900/1*_8FqWVJDL3kfL_GWjKe4OA.png)

Nightly prices for each check-in date on the calendar as set by Smart Pricing

Learning the booking lead time distribution helps power our pricing system. Airbnb launched Smart Pricing to help hosts set optimal prices and maximize earnings. These tools take into account factors like demand, supply and individual listing properties in order to make price suggestions for all check-in dates on the calendar. However, market conditions typically change as booking dates get closer to check-in dates, and, as a result, it is critical for us to account for these changes and help hosts keep their prices optimized for market conditions.

As an example, on a high demand night like New Year’s Eve, guests tend to book more in advance (i.e. at long lead times) than at other times of the year. This information helps set the right prices for New Year’s Eve. Similarly, locations play a big part in this too. A supply-constrained market gets bookings well ahead of check-ins compared to a holiday market like South Beach, Miami. By learning the arrival process for every check-in date and location, Smart Pricing accounts for this “early demand” and generates a pricing policy that allows hosts to optimally update their prices as we approach check-in.

#### <u>What Does the Arrival Process Look Like?</u>

To make the problem more concrete, let’s start by introducing some notation. Let *X_T(t) = P( Xijt=1 | Bij = 1)* represent the lead time distribution of guest bookings, where

- *T* is a <u>random variable</u> representing lead time
- *i* be check-in date
- *j* be a listing of interest
- ~~*Bij* be 1 if the check-in day ended up getting booked~~
- ~~*Xijt* be 1 if the check-in day was booked at lead time~~

The figure below is an example of the lead time distribution aggregated to a market level. As we can see, the booking mass density gradually increases as we approach check-in (right to left), since guests don’t always plan their trips well in advance.

![img](https://cdn-images-1.medium.com/max/1200/0*laQ8afu4aUvB4Ogj)

Our goal is to learn and estimate *f* (the density of the distribution above), for each listing *i*, check-in date *j*, and every lead day *t* heading into check-in.

### Can We Use Machine Learning Directly?

What would using a typical ML approach look like for this problem? Well, we would first build out training dataset with relevant features X and labels Y. In our case, the label would be the lead time for every guest booking. To predict this label, Airbnb has accumulated various predictor signals that capture market supply, market demand, as well as listing-level features. The model would then predict the average lead time for each check-in date. However, in the end, we concluded that an ML approach would have several complications:

- **Accounting for probabilistic outcomes**: The arrival process is stochastic. For the purposes of optimal pricing, we need to know the distribution of bookings over lead time and not just the average lead time to book.
- **Sparsity**: Every listing gets booked at most once for a check-in date, leading to a highly sparse data set that would be challenging to handle without adding significant model complexity We will have to architect the model in a way that pools listing information together, enabling transfer learning across listings.
- **High dimensionality**: At Airbnb, we have millions of unique listings — each with its own defining characteristics and features that govern the arrival process. This makes every listing a very high dimensional data point and inefficient to use in a classic ML framework.
- **Scale**: The model will need to make predictions along three dimensions, ie. the triplet of (listings x check-in x number of lead times). This comes at a *O(10⁶ x 10² x 10³)* complexity, requiring very large training and scoring data sets and risks poor latency.

In addition to these challenges, close examination of the lead time distribution revealed a distinct structure of unimodal distribution with clear cyclical patterns (likely weekday/weekend movements) — in fact, taking a closer look, we noticed that the distributions resembled a generalized exponential family. Considering these strong parametric characteristics along with the complexities of an ML-only approach we were inspired to try a hybrid approach combining ML and structural modeling.

### Machine Learning vs Structural Modeling, or Both?

Modern ML models fare very well in terms of predictive performance, but seldom model the underlying data generation mechanism. In contrast, structural models provide interpretability by allowing us to explicitly specify the relationships between the variables (features and responses) to reflect the process that gives rise to the data, but often fall short on predictive performance. Combining the two schools of thought allows us to exploit the strengths of each approach to better model the data generating process as well as achieve good model performance.

When we have good intuition for a modeling task, we can use our insights to reinforce an ML model with ~~structural context~~. Imagine we are looking to predict a response *Y* based on features *(X₀,…,Xn). O*rdinarily, we would train our favorite ML model to predict. However, suppose we also know that *Y is* distributed over an input feature *X₀* with a distribution *F* ~~parameterized~~ by 𝜃 i.e. *Y~ F(X₀; 𝜃 ),* we could leverage this information and decompose the task to learning 𝜃 using features *(X₀,…,Xn)*, and then simply plug our estimate of 𝜃 back into f to arrive at *Y* in the final step.

By employing this hybrid approach, we can leverage both the algorithmic powerhouse that ML provides and the informed intuition of statistical modeling. This is the approach we took to model lead time dynamics.

### The Modeling Methodology

#### 1. Laying the Foundation — Generating Demand Aggregations

As a first step, we start by pooling our supply to form listing clusters. The clustering is learned using guest search patterns on Airbnb, with each cluster mapped to a common set of guest preferences. As a result, the listings within a cluster share common demand profiles and tend to witness similar lead time distributions. This process helps overcome problems of dimensionality and scale.

Unlike commoditized accommodations in the hotel industry, every listing on Airbnb is unique. Airbnb homes span over a broad spectrum, from price, location, quality, to size, etc. This vast heterogeneity brings challenges for personalization. It’s challenging to estimate the arrival process for every listing and every check-in date separately. Clustering listings into “demand aggregations” addresses this challenge. A demand aggregation refers to a cluster of listings that share common demand profiles.

When guests come to Airbnb, they browse through multiple listings in our catalogue and potentially choose one to book. By tracing a guest’s *path to purchase,* we learn more about their considerations and preferences. When two listings frequently co-appear in the purchase path of several guests, they tend to mirror a common set of guest preferences. For example, listings in west lake Tahoe often co-appear in search sessions of ski enthusiasts looking to stay at Airbnb (picture below).

We can employ the notion above to generate low dimensional listing embeddings, similar to [this embedding technique](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e). This work produces listing embeddings which semantically relate to their demand profiles. With the listing embeddings at hand, we then clustered them together hierarchically to form demand aggregations. Each demand aggregation represents a *consideration set* that guests browsed over when making a booking. Such a framework is quite common in discrete choice models for demand estimation, specially online marketplaces. Here are some visuals of how these clusters might looks like.

![img](https://cdn-images-1.medium.com/max/1500/0*etJhzm9TEs2lSYYM)

Listing clusters map to identifiable guest preferences

This supply pooling step effectively reduces the cardinality on the listing dimension and helps address some of the problems enumerated above.

#### 2. Adding Reinforcements — Using Structural Modeling

With clusters serving as a foundation, we can build a structurally reinforced ML model by having the lead time distribution adhere to a parameterized functional form. This helps us bypass having to make predictions for each lead time *t* and instead we predict the distribution by simply predicting the parameters of the functional form. This reduces the dimension of the prediction to just a couple of parameters.

If we make a simple assumption that the number of bookings in a unit interval of time is Poisson, then the time to event can be approximated by a gamma distribution. In other words, we can explicitly model the relationship between booking density and lead time by assuming the lead time distribution per cluster to be Gamma with unknown parameters *⍺, β.*

![img](https://cdn-images-1.medium.com/max/900/0*uZn1NwrHqIy8GNR1)

This actually agrees with the observation we made earlier about the lead time distribution resembling the exponential family. This approach also has a nice ~~Bayesian~~ interpretation since the Gamma is the conjugate prior for the parameter in a Poisson process. The lead time distribution that results, will have the following probability density and cumulative distribution function:

With this structure in place, It becomes sufficient to predict the two parameters *⍺, β* and use them in the distribution’s functional form to generate the density at each lead time.

Similarly, we alter the functional form to account for cyclical patterns too. We treat the arrival time series as an waveform and decompose it to extract the inherent components. This typically involves,

1. ~~De-trending~~ the time series to work on the residuals
2. Applying a ~~Fourier Transform~~ to get the resonant frequency (⍵)
3. Determining the amplitude (𝜌) and the phase angle (*φ)*

As as result, we obtain the stationary oscillating waveform *f_c(t) = 𝜌.sin(⍵.t + φ)* that models the cyclic component. This can be extended to as many harmonics as needed based on the use case. Finally, we combine *f_c(t)* with *f_b(t)* to get the final functional form to characterize the arrival process

This form has 5 parameters *(⍺, β, ⍵, 𝜌, φ)*. To predict the lead time distribution for every future check-in date, we only need to predict these 5 values.

#### 3. The Final Piece of the Puzzle — Forecasting the Parameters

We start by building the training data with available ~~predictors~~ (*X*) and the empirical lead time distribution as the label (plot below). We then train our ML model to find the best parameter set *(⍺, β, ⍵, 𝜌, φ)* that maximizes likelihood given the training set. The resulting parameter estimates will best approximate the functional form *f_T(t)* to the observed empirical distribution. The plots below demonstrate this

![img](https://cdn-images-1.medium.com/max/1500/1*jZeNNlogXt8ZIwsjgzhfDQ.png)

Predicting the empirical lead time distribution using the ML + Structural modeling framework (a) Mass Density (b) Cumulative Density

Using this framework of modeling has several advantages. Notably we

- **Reduce problem complexity:** In our case, we do so by reducing the dimensions of listing and lead time.
- **Add interpretability:** The various parameters of the structural model informs us about different components of the arrival process.
- **Prevent overfitting:** In this example, enforcing a parametric form to the model output provides additional information during model training which in turn serves as a natural regularizer.

This model is currently used in product, primarily to power Smart Pricing. The tool uses the predicted lead time distributions for each check-in to help hosts keep prices up to date. We also use it to inform hosts about booking lead times statistics to help them make informed decisions around calendar availability. For instance, letting hosts in Tahoe know that we expect guests to be booking well in advance due to an upcoming ski season can help them unblock dates on their calendars early enough to capture the demand.

![img](https://cdn-images-1.medium.com/max/1200/0*GoGa2VD-fKjHymzq)

**Product use cases:** **(a) Guest side:** Information about average lead times for LA, helps guests time their booking. **(b) Host side:** Knowing the percentage of completed bookings can help drive pricing decisions.

### Final Thoughts

Using the structural + ML framework in this project helped halve the error and generalized better than the conventional ML algorithms that we tried. It also helped us understand and systematically model lead time dynamics. The framework is applicable to many other modeling scenarios as well, specifically when

- Imposing an explicit relationship between a predictor(s) and the response
- Isolating effects of critical variables of interest, such as a treatment or a risk factor
- Modeling the data generation mechanism for a stochastic process
- Dealing with imperfect training data, e.g. dealing with censored or truncated data

Today, machine learning has become a common tool of a data scientist’s toolkit. We often treat ML like a silver bullet - using off-the-shelf methods to build out data products. There are also a number of open source tools and that help automate various aspects of model development like feature generation, model selection, deployment etc. making model building much easier.

In such a world where the procedural details of model development have been abstracted away, sound problem formulation gains prominence. In particular, we can use domain knowledge to equip these ML models with the right structure and formulation using structural techniques. Utilizing human product intuition to architect, guide, and augment ML models in such a manner can truly push the boundaries of model performance and interpretability.

------

Want to learning more about market dynamics and build such frameworks? We’re always looking for [talented people to join our Data Science team](https://www.airbnb.com/careers/departments/data-science-analytics)!

------

Special thanks to all my team members who contributed to this project and helped review this post. Milan Shen, Rubel Lobel, Minyong Lee, Chen-Hung Wu, Bar Ifrach, Robert Chang, Navin Sivanandam, Jeff Feng and Clara Lam