from vector import retriever

# testing retriever call

def test_retriever(question):
    print(retriever.invoke(question))

if __name__ == "__main__":
    test_retriever('what is the best review')