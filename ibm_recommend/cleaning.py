def email_mapper(email):
    '''
    Maps email to a unique id

    INPUT:
    email: hash string
    '''

    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in email:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded
