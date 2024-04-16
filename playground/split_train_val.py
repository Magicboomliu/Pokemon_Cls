import os
import random


def split_train_val(list,percentage=0.1):
    # Calculate 10% of the list length
    sample_size = round(len(list) * 0.1)

    # Take a random sample of 10% of the elements
    random_sample = random.sample(list, sample_size)

    # Get the remaining elements that are not in the random sample
    remaining_elements = [element for element in list if element not in random_sample]
    
    
    return remaining_elements, random_sample
    



if __name__=="__main__":
    
    
    datapath = "/data1/su/pokemon/pokemon_dataset/"

    
    # 14 types pokemons
    pokemon_category_dict = os.listdir(datapath)
    
    train_list = []
    val_list = []
    
    for idx, pokemon_type in enumerate(pokemon_category_dict):
        
        pokemon_sub_folder = os.path.join(datapath,pokemon_type)
        
        instances_nums_of_pokemon = len(os.listdir(pokemon_sub_folder))
        
        train_list_current,val_list_current = split_train_val(os.listdir(pokemon_sub_folder),0.1)
        
        train_list_current = [os.path.join(pokemon_type,f) for f in train_list_current]
        val_list_current = [os.path.join(pokemon_type,f) for f in val_list_current]
        
        train_list.extend(train_list_current)
        val_list.extend(val_list_current)
        
        

    with open("pokemon_train.txt",'w') as f:
        for idx, fname in enumerate(train_list):
            if idx!=len(train_list)-1:
                f.writelines(fname+"\n")
            else:
                f.writelines(fname)

    with open("pokemon_val.txt",'w') as f:
        for idx, fname in enumerate(val_list):
            if idx!=len(val_list)-1:
                f.writelines(fname+"\n")
            else:
                f.writelines(fname)
        
        
    

    
    
    
    pass