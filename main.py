def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--process', type=bool, default=False)
    parser.add_argument('--modeltype', type=str, default='mf')
    parser.add_argument('--f_dim', type=int, default=32)
    parser.add_argument('--pc', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_epoch', type=int, default=20)
    args = parser.parse_args()
    
    # import dataset handler functions
    from modules import ds_handler
        
    # download dataset
    if args.download == True:
        ds_handler.download_data()

    # preprocess dataset    
    if args.process == True:
        ds_handler.split_save_data()
        ds_handler.save_label_encoder()
    
    # load preprocessed dataset and etc.
    train_set, val_set, test_set, user_enc, movie_enc =\
            ds_handler.load_mf()
    
    # import model class, functions
    if args.modeltype == 'mf':
        from modules import mf
    
    # define model
    n_u, n_m = len(user_enc), len(movie_enc)
    f_dim = args.f_dim
    train_avg = mf.calc_avg(train_set, user_enc, movie_enc)
    model = mf.MF(n_u, n_m, f_dim, train_avg)
    
    # train model
    pc, lr, max_epoch = args.pc, args.lr, args.max_epoch
    model = mf.train(model, train_set, val_set,
            user_enc, movie_enc, pc, lr, max_epoch)
    
    # print rmse and export result file
    mf.make_result(model, train_set, test_set, user_enc, movie_enc)

if __name__ == '__main__':
    main()