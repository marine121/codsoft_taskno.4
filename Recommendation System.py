import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data: books with genres and descriptions
data = {
    'BookID': [1, 2, 3, 4, 5],
    'Title': ['Book A', 'Book B', 'Book C', 'Book D', 'Book E'],
    'Genre': ['Fiction', 'Non-Fiction', 'Fiction', 'Science', 'Fiction'],
    'Description': [
        'A thrilling fiction novel with adventure and mystery.',
        'A comprehensive guide to modern science and technology.',
        'An epic fiction story set in a dystopian future.',
        'An informative book on the latest scientific discoveries.',
        'A fiction book about a detective solving crimes.'
    ]
}

books_df = pd.DataFrame(data)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['Description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_books(title, cosine_sim=cosine_sim):
    idx = books_df.index[books_df['Title'] == title].tolist()[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    book_indices = [i[0] for i in sim_scores[1:]]  # Exclude the book itself

    # Return the top 3 similar books
    return books_df['Title'].iloc[book_indices].tolist()

# Example usage
print(recommend_books('Book A'))
