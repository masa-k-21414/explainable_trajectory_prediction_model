from tensorboardX import SummaryWriter
def tensorboard(writer, epoch, epoch_d_loss=False, epoch_g_loss=False, epoch_g_l2=False, 
                epoch_val_ade=False, epoch_val_adenl=False, epoch_train_ade=False, epoch_train_adenl=False, epoch_g_at=False, new_loss=False):

    if epoch_val_ade != False:
        writer.add_scalar('v_ade', epoch_val_ade, epoch)
    if epoch_val_adenl != False:
        writer.add_scalar('v_adenl', epoch_val_adenl, epoch)
    if epoch_train_ade != False:
        writer.add_scalar('t_ade', epoch_train_ade, epoch)
    if epoch_train_adenl != False:
        writer.add_scalar('t_adenl', epoch_train_adenl, epoch)
    if epoch_g_l2 != False:
        writer.add_scalar('g_mse', epoch_g_l2, epoch)
    if epoch_d_loss != False:
        writer.add_scalar('d_loss', epoch_d_loss, epoch)
    if epoch_g_loss != False:
        writer.add_scalar('g_loss', epoch_g_loss, epoch)
    if epoch_g_at != False:
        writer.add_scalar('g_at', epoch_g_at, epoch)
    if new_loss != False:
        writer.add_scalar('new loss', new_loss, epoch)
