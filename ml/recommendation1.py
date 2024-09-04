
# %%
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import surprise
import sklearn
from controllers.db import get_db_connection

# %%
# wishlist_data = {
#     'wishlist_id': list(range(1, 41)),  # Expanded to 40 entries
#     'user_id': [
#         101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104,
#         105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108,
#         109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112,
#         113, 113, 113, 114
#     ],  # Each user has multiple wishlist items
#     'book_id': [
#         201, 202, 203, 202, 203, 204, 203, 204, 205, 204, 205, 206,
#         205, 206, 207, 206, 207, 208, 207, 208, 209, 208, 209, 210,
#         209, 210, 211, 210, 211, 212, 211, 212, 213, 212, 213, 214,
#         213, 214, 215, 215
#     ],
#     'radius': [
#         5.0, 7.0, 15.0, 10.0, 12.0, 20.0, 15.0, 17.0, 25.0, 7.0, 12.0, 20.0,
#         14.0, 18.0, 22.0, 11.0, 15.0, 24.0, 6.0, 10.0, 19.0, 8.0, 14.0, 21.0,
#         13.0, 17.0, 23.0, 9.0, 15.0, 20.0, 12.0, 18.0, 24.0, 16.0, 20.0, 25.0,
#         14.0, 19.0, 23.0, 21.0
#     ]
# }


# %%
# ratings_data = {
#     'ratings_id': list(range(1, 21)),
#     'user_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
#     'book_id': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220],
#     'rating': [4, 5, 3, 4, 2, 5, 3, 4, 1, 5, 2, 3, 4, 5, 1, 2, 4, 3, 5, 2]
# }

# %%
# books_data = {
#     'book_id': list(range(201, 221)),
#     'title': [
#         'To Kill a Mockingbird', '1984', 'The Great Gatsby', 'The Catcher in the Rye', 'The Hobbit',
#         'Fahrenheit 451', 'Brave New World', 'Moby-Dick', 'War and Peace', 'Pride and Prejudice',
#         'The Lord of the Rings', 'Jane Eyre', 'The Diary of a Young Girl', 'Crime and Punishment', 'Wuthering Heights',
#         'The Grapes of Wrath', 'Catch-22', 'The Odyssey', 'Little Women', 'The Picture of Dorian Gray'
#     ],
#     'authors': [
#         'Harper Lee', 'George Orwell', 'F. Scott Fitzgerald', 'J.D. Salinger', 'J.R.R. Tolkien',
#         'Ray Bradbury', 'Aldous Huxley', 'Herman Melville', 'Leo Tolstoy', 'Jane Austen',
#         'J.R.R. Tolkien', 'Charlotte Brontë', 'Anne Frank', 'Fyodor Dostoevsky', 'Emily Brontë',
#         'John Steinbeck', 'Joseph Heller', 'Homer', 'Louisa May Alcott', 'Oscar Wilde'
#     ],
#     'categories': [
#         'Fiction', 'Dystopian', 'Classic', 'Classic', 'Fantasy',
#         'Dystopian', 'Dystopian', 'Classic', 'Historical', 'Classic',
#         'Fantasy', 'Classic', 'Autobiography', 'Classic', 'Classic',
#         'Classic', 'Satire', 'Epic', 'Classic', 'Classic'
#     ],
#     'lang': ['English'] * 20,
#     'isbn': [
#         '9780061120084', '9780451524935', '9780743273565', '9780316769488', '9780345339683',
#         '9781451673319', '9780060850524', '9781503280786', '9781400079988', '9780141439518',
#         '9780618640157', '9780142437209', '9780553296983', '9780140449136', '9781853262959',
#         '9780143039433', '9780684833392', '9780140268867', '9780316055437', '9780141439556'
#     ],
#     'image': [
#         'image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg',
#         'image6.jpg', 'image7.jpg', 'image8.jpg', 'image9.jpg', 'image10.jpg',
#         'image11.jpg', 'image12.jpg', 'image13.jpg', 'image14.jpg', 'image15.jpg',
#         'image16.jpg', 'image17.jpg', 'image18.jpg', 'image19.jpg', 'image20.jpg'
#     ]
# }

# %%
# users_data = {
#     'user_id': list(range(101, 121)),
#     'email': [
#         'user1@example.com', 'user2@example.com', 'user3@example.com', 'user4@example.com', 'user5@example.com',
#         'user6@example.com', 'user7@example.com', 'user8@example.com', 'user9@example.com', 'user10@example.com',
#         'user11@example.com', 'user12@example.com', 'user13@example.com', 'user14@example.com', 'user15@example.com',
#         'user16@example.com', 'user17@example.com', 'user18@example.com', 'user19@example.com', 'user20@example.com'
#     ],
#     'first_name': [
#         'John', 'Jane', 'Jim', 'Jack', 'Jill',
#         'Joe', 'Jenny', 'Jacob', 'Jasmine', 'James',
#         'Julia', 'Jordan', 'Jared', 'Joan', 'Jerry',
#         'Jesse', 'Jorge', 'Joanna', 'Joy', 'Jake'
#     ],
#     'last_name': [
#         'Doe', 'Smith', 'Brown', 'Johnson', 'Williams',
#         'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez',
#         'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson',
#         'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson'
#     ],
#     'password': [
#         'password1', 'password2', 'password3', 'password4', 'password5',
#         'password6', 'password7', 'password8', 'password9', 'password10',
#         'password11', 'password12', 'password13', 'password14', 'password15',
#         'password16', 'password17', 'password18', 'password19', 'password20'
#     ],
#     'lat': [
#         37.7749, 34.0522, 40.7128, 41.8781, 29.7604,
#         39.0997, 32.7767, 36.1699, 47.6062, 25.7617,
#         42.3601, 37.3382, 39.7392, 38.9072, 37.7749,
#         34.0522, 40.7128, 41.8781, 29.7604, 39.0997
#     ],
#     'lng': [
#         -122.4194, -118.2437, -74.0060, -87.6298, -95.3698,
#         -94.5786, -96.7970, -115.1398, -122.3321, -80.1918,
#         -71.0589, -121.8863, -104.9903, -77.0369, -122.4194,
#         -118.2437, -74.0060, -87.6298, -95.3698, -94.5786
#     ]
# }


# %%
def fetch_data_from_db():
    conn = get_db_connection()
    wishlist = conn.table('wishlists').select('*').execute()
    ratings =conn.table('books_ratings').select('*').execute()
    books = conn.table('books').select('*').execute()
    return wishlist, ratings, books

# %%
wishlist, ratings, books = fetch_data_from_db()

# %%
# wishlist_df=pd.DataFrame(wishlist_data)
# ratings_df = pd.DataFrame(ratings_data)
# books_df = pd.DataFrame(books_data)
# users_df = pd.DataFrame(users_data)

# %%
# ratings_with_titles= pd.merge(ratings_df,books_df, on='book_id')
# ratings_with_titles

# %%
# wishlist_df.to_csv('wishlist.csv', index=False)
# ratings_df.to_csv('ratings.csv', index=False)
# books_df.to_csv('books.csv', index=False)
# users_df.to_csv('users.csv', index=False)

# %%
# Convert to DataFrame
wishlist_df = pd.DataFrame(wishlist.data)
ratings_df = pd.DataFrame(ratings.data)
books_df=pd.DataFrame(books.data)

wishlist_ratings_df = pd.merge(wishlist_df[['user_id', 'book_id']], ratings_df, on=['user_id', 'book_id'], how='inner')

# Combine the wishlist ratings with the rest of the ratings
combined_df = pd.concat([wishlist_ratings_df, ratings_df]).drop_duplicates().reset_index(drop=True)

# Ensure there are no duplicate entries
combined_df.drop_duplicates(subset=['user_id', 'book_id'], keep='first', inplace=True)
combined_df

# %%
#Applying Collaborative Filtering
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy

# %%
# Define the Reader object with the correct rating scale
reader = Reader(rating_scale=(1, 5))

# Load the combined data into Surprise's Dataset format
data = Dataset.load_from_df(combined_df[['user_id', 'book_id', 'rating']], reader)
data

# %%
# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# %%
#SVD (Singular Value Decomposition) is a matrix factorization technique commonly used in recommendation systems. 
# Define the SVD model
model = SVD()

# Train the model
model.fit(trainset)

# Test the model
predictions = model.test(testset)

# Calculate and print the accuracy metrics
accuracy.rmse(predictions)


# %%
#Prediction with KNN Model

from surprise import KNNBasic

sim_options = {
    'name': 'cosine',
    'user_based': True
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)
predictions = model.test(testset)
print("KNN RMSE:", accuracy.rmse(predictions))


# %%
from surprise.model_selection import GridSearchCV

param_grid = {
    'n_factors': [50, 100], #Number of hidden attributes used to understand user preferences and item characteristics.
    'n_epochs': [20, 30], #Number of times the model practices (or trains) on the data.
    'lr_all': [0.002, 0.005], #It’s like adjusting the speed at which you’re learning from your mistakes. 
    'reg_all': [0.2, 0.4] #Helps to prevent the model from becoming too complex and fitting the training data too closely.
}

grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)
print("Best parameters using GridSearchCV :", grid_search.best_params['rmse'])
print("Best RMSE score GridSearchCV:", grid_search.best_score['rmse'])

# Extract the best model
best_model = grid_search.best_estimator['rmse']

# Train the best model on the full training set
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Evaluate the best model using the existing test set
predictions = best_model.test(testset)
print("Test set RMSE:", accuracy.rmse(predictions))


# %%
# Define a function to get book recommendations for a specific user
def get_book_recommendations(user_id, model, books_df, num_recommendations=5):
    # Get the list of all book IDs
    all_book_ids = books_df['book_id'].unique()
    
    # Generate predictions for all books for the given user
    predictions = [model.predict(user_id, book_id) for book_id in all_book_ids]
    
    # Sort predictions by estimated rating
    sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)
    
    # Get top N book IDs
    top_predictions = sorted_predictions[:num_recommendations]
    
    # Extract book IDs and estimated ratings
    recommended_book_ids = [pred.iid for pred in top_predictions]
    recommended_ratings = [pred.est for pred in top_predictions]
    
    # Merge with book details
    recommended_books = books_df[books_df['book_id'].isin(recommended_book_ids)]
    recommended_books = recommended_books.copy()
    recommended_books['predicted_rating'] = recommended_ratings
    
    return recommended_books[['book_id', 'title', 'authors', 'predicted_rating']]

user_id = 120
recommended_books = get_book_recommendations(user_id, best_model, books_df)
print(recommended_books)

# %%
#recommendation for wishlist
user_item_matrix = wishlist_df.pivot_table(index='user_id', columns='book_id', aggfunc='size', fill_value=0)
user_item_matrix

# %%
from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


# %%
def recommend_books_based_on_wishlist(user_id, wishlist_df, ratings_df, books_df, num_recommendations=5):
    # Get the list of books in the current user's wishlist
    user_wishlist_books = wishlist_df[wishlist_df['user_id'] == user_id]['book_id'].tolist()
    
    # Find other users who have similar books in their wishlist
    similar_users = wishlist_df[wishlist_df['book_id'].isin(user_wishlist_books)]['user_id'].unique()
    
    if len(similar_users) == 0:
        print(f"No similar users found for user {user_id}.")
        return pd.DataFrame(columns=['book_id', 'title', 'authors'])

    # Aggregate the books from these similar users' wishlists, excluding the books already in the target user's wishlist
    similar_users_books = wishlist_df[(wishlist_df['user_id'].isin(similar_users)) & (~wishlist_df['book_id'].isin(user_wishlist_books))]

    # If no books found in similar users' wishlists, return an empty DataFrame
    if similar_users_books.empty:
        print(f"No additional books found in similar users' wishlists for user {user_id}.")
        return pd.DataFrame(columns=['book_id', 'title', 'authors'])
    
    # Count how often each book appears in the similar users' wishlists
    book_recommendations = similar_users_books['book_id'].value_counts().head(num_recommendations)
    
    # Get the book details for these recommended books
    recommended_books_df = books_df[books_df['book_id'].isin(book_recommendations.index)]
    
    return recommended_books_df

# Example usage
# user_id = 58  # Replace with the actual user_id
recommended_books = recommend_books_based_on_wishlist(user_id, wishlist_df, ratings_df, books_df)
print(recommended_books)


# %%



