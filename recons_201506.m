% dir = '/Volumes/Macintosh HD/Users/leeneil/Documents/MATLAB/sacla201412/';
dir = 'G:\CDI_data\201506_sacla';
% close all;
run = 334911;
dect = 1;
tag = 227;
par = 0;
% 259962: 039, 150, 268, 291, 471, 574, 309   (BKG: 329)
% 259965: 032, 053, 414
% 259967: 247, 280, 304, 484, 546
% 259970: 116, 255, 282
% 260081: 081, 093

grp = floor( (tag - 1) / 72 );
sub_tag = tag - 72*grp;


% info = hdf5info([dir '\Bkg\run' int2str(run)  '\ReduceData\BG_AVG.h5']);
% bkg_path = info.GroupHierarchy.Groups(2).Groups(1).Groups(1).Datasets;
% bkg = hdf5read(bkg_path.Filename, bkg_path.Name);
bkg = h5value([dir '\Bkg\run' int2str(run)  '\ReduceData\BG_AVG.h5'], [2 1 1 1]);

% info = hdf5info([dir '\Data\run' int2str(run)  '\MapData\' int2str(run) '_' int2str(grp) '.h5']);
% dect1_path = info.GroupHierarchy.Groups(2).Groups(1).Groups(1+sub_tag).Datasets(1);
% img = hdf5read(dect1_path.Filename, dect1_path.Name);
img = h5value([dir '\Data\run' int2str(run)  '\MapData\' int2str(run) '_' int2str(grp) '.h5'],...
    [2 1 (sub_tag+1) 1]);

% dect2_path = info.GroupHierarchy.Groups(2).Groups(2).Groups(1+sub_tag).Datasets(1);
% img_dual = hdf5read(dect1_path.Filename, dect2_path.Name);



% img1 = img;
% bkg( bkg < 0) = 0;
img1 = img - bkg;

dtr_gain = h5value([dir '\Data\run' int2str(run)  '\MapData\' int2str(run) '_' int2str(grp) '.h5'], [2 1 1 1]);

img2 = dtr_gain / (4000/3.65) * img1;
save([dir '\' int2str(run) '_' int2str(tag)], 'img2');
img2( img2 < 1) = 0;
% img3 = CDIpreprocess(img2, 3, 200, 'SACLA201412', [7 -1]);
[img3, dx] = CDIpreprocess(img2, 7, 100, 'SACLA201506', []);

img3( img3 < 0) = 0;

% img3( 112:124, 112:124 ) = 0;
% img3( 138:158, 138:158) = 0;
% img3(130:146, 132:144) = 0;
% img3(84:95, 84:95) = 0;

load jet2;

figure(102);
imagesc( log10( img3 ) );
axis image;
colorbar;
title(['tag #' int0str(tag, 3)]);
colormap(jet2);

beta1 = 5.348791537873313e-16;


mkdir([dir '\recon_log\' int2str(run)]);
mkdir([dir '\recon_log\' int2str(run) '\' int0str(tag,3 )]);
%% GSW



% 1: GHIO
% 2: GSW
% 3: HIO
% 4: SW
% 5: GHIO+GSW
% 6: GSW+GHIO
% 7: GHIO (circular support)
% 8: GHIO+GSW (circular support)
% 9: Customized
recon_recipe = 8;
% GHIO
support_sz = 30;
n_itrs = 2000;
n_gen1 = 10;
n_cpys = 8;
% GSW
n_itrs1 = 50;
n_itrs2 = 50;
n_gen2 = 20;
n_conv1 = 3;
n_conv2 = 1.5;
cutoff1 = 0.03;
cutoff2 = 0.08;
% Custumized
pat_cutoff = 0.15;


datetime = clock;
datetime_string = [int2str(datetime(1)) int0str(datetime(2), 2) int0str(datetime(3), 2)...
    int0str(datetime(4), 2) int0str(datetime(5), 2) int0str(floor(datetime(6)), 2)];
fid = fopen([dir '\recon_log\' int2str(run) '\' int0str(tag,3 ) '\' datetime_string '.txt'], 'w+');
fprintf(fid, [ datetime_string '\n']);
switch recon_recipe
     case 1,
        fprintf(fid, 'Method: GHIO\n');
        fprintf(fid, ['Support size: ' int2str(support_sz) '\n']);
        fprintf(fid, ['Iteration #: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Generation # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Copy #: ' int2str(n_cpys) '\n']);
    
    case 2,
        fprintf(fid, 'Method: Guided-SW\n');
        fprintf(fid, ['Copy #: ' int2str(n_cpys) '\n']);        
        fprintf(fid, ['Iteration # 1: ' int2str(n_itrs1) '\n']);
        fprintf(fid, ['Iteration # 1: ' int2str(n_itrs2) '\n']);
        fprintf(fid, ['Generation # 2: ' int2str(n_gen2) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv1) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv2) '\n']);
        fprintf(fid, ['cutoff 1: ' num2str(cutoff1) '\n']);
        fprintf(fid, ['cutoff 2: ' num2str(cutoff2) '\n']);
    
    case 5,
        fprintf(fid, 'Method: GHIO+GSW\n');
        fprintf(fid, ['Support size: ' int2str(support_sz) '\n']);
        fprintf(fid, ['Iteration #: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Generation # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Copy #: ' int2str(n_cpys) '\n']);
        
        fprintf(fid, ['Iteration # 1: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Iteration # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Generation # 2: ' int2str(n_gen2) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv1) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv2) '\n']);
        fprintf(fid, ['cutoff 1: ' num2str(cutoff1) '\n']);
        fprintf(fid, ['cutoff 2: ' num2str(cutoff2) '\n']);
    case 6,
        fprintf(fid, 'Method: GSW+GHIO\n');
        fprintf(fid, ['Iteration # 1: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Iteration # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Generation # 2: ' int2str(n_gen2) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv1) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv2) '\n']);
        fprintf(fid, ['cutoff 1: ' num2str(cutoff1) '\n']);
        fprintf(fid, ['cutoff 2: ' num2str(cutoff2) '\n']);    
        fprintf(fid, ['Iteration #: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Generation # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Copy #: ' int2str(n_cpys) '\n']);
    case 7,
        fprintf(fid, 'Method: GHIO (circular support)\n');
        fprintf(fid, ['Support size: ' int2str(support_sz) ' (diameter)\n']);
        fprintf(fid, ['Iteration #: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Generation # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Copy #: ' int2str(n_cpys) '\n']);
    case 8,
        fprintf(fid, 'Method: GHIO+GSW (circular)\n');
        fprintf(fid, ['Support size: ' int2str(support_sz) '\n']);
        fprintf(fid, ['Iteration #: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Generation # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Copy #: ' int2str(n_cpys) '\n']);
        
        fprintf(fid, ['Iteration # 1: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Iteration # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Generation # 2: ' int2str(n_gen2) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv1) '\n']);
        fprintf(fid, ['Convolution size 1: ' num2str(n_conv2) '\n']);
        fprintf(fid, ['cutoff 1: ' num2str(cutoff1) '\n']);
        fprintf(fid, ['cutoff 2: ' num2str(cutoff2) '\n']);
    case 9,
        fprintf(fid, 'Method: Customized\n');
        fprintf(fid, ['Iteration #: ' int2str(n_itrs) '\n']);
        fprintf(fid, ['Generation # 1: ' int2str(n_gen1) '\n']);
        fprintf(fid, ['Copy #: ' int2str(n_cpys) '\n']);
        fprintf(fid, ['Patterson cutoff: ' int2str(pat_cutoff) '\n']);
    otherwise,
        fprintf(fid, 'no reconstruction recipe specified\n');
end
fclose(fid);

switch recon_recipe
    case 1,
        Sup = hiosupport(length(img3), support_sz);
        rs = ghio2d( ifftshift(sqrt(img3)), Sup, n_itrs, n_gen1, n_cpys,  ifftshift(img3==0), [] );
        pat = fftshift(ifft2( ifftshift(beta1*abs( fftshift( ifft2(rs(:,:,end) ) ) ).^2 ), 'symmetric'));
        pat = pat ./ max(pat(:));
    case 2,
        [rs, Sup] = gshrinkwrap( ifftshift(sqrt(img3)), n_itrs1, ifftshift(img3==0), n_gen2, n_itrs2, n_cpys, [], n_conv1, cutoff1, cutoff2);
    case 5,
        Sup = hiosupport(length(img3), support_sz);
        rs0 = ghio2d( ifftshift(sqrt(img3)), Sup, n_itrs, n_gen1, n_cpys,  ifftshift(img3==0), [] );
        [rs, Sup]  = gshrinkwrap( fft2(rs0(:,:,end)), n_itrs1, ifftshift(img3==0), n_gen2, n_itrs2, n_cpys, [], n_conv1, cutoff1, cutoff2);
    case 6,
        [rs, Sup] = gshrinkwrap( ifftshift(sqrt(img3)), n_itrs1, ifftshift(img3==0), n_gen2, n_itrs2, n_cpys, [], n_conv1, cutoff1, cutoff2);
        [rs, G] = ghio2d( fft2(rs(:,:,end)), Sup(:,:,end), n_itrs, n_gen1, n_cpys,  ifftshift(img3==0), [] )  
    case 7,
        Sup = circle( zeros(support_sz) );
        if mod(size(Sup,1), 2) == mod(size(img3,1), 2)
            sup_pad = ( size(img3,1) - size(Sup,1) ) / 2;
            Sup = padarray(Sup, [sup_pad sup_pad]);
            Sup = Sup > 0;
        else
            sup_pad = ( size(img3,1) + 1 - size(Sup,1) ) / 2;
            Sup = padarray(Sup, [sup_pad sup_pad]);
            Sup = Sup(2:end, 2:end);
            Sup = Sup > 0;
        end
        rs = ghio2d( ifftshift(sqrt(img3)), Sup, n_itrs, n_gen1, n_cpys,  ifftshift(img3==0), [] );
        pat = fftshift(ifft2( ifftshift(beta1*abs( fftshift( ifft2(rs(:,:,end) ) ) ).^2 ), 'symmetric'));
        pat = pat ./ max(pat(:));
    case 8,
        Sup = circle( zeros(support_sz) );
        if mod(size(Sup,1), 2) == mod(size(img3,1), 2)
            sup_pad = ( size(img3,1) - size(Sup,1) ) / 2;
            Sup = padarray(Sup, [sup_pad sup_pad]);
            Sup = Sup > 0;
        else
            sup_pad = ( size(img3,1) + 1 - size(Sup,1) ) / 2;
            Sup = padarray(Sup, [sup_pad sup_pad]);
            Sup = Sup(2:end, 2:end);
            Sup = Sup > 0;
        end
        rs0 = ghio2d( ifftshift(sqrt(img3)), Sup, n_itrs, n_gen1, n_cpys,  ifftshift(img3==0), [] );
        [rs, Sup]  = gshrinkwrap( fft2(rs0(:,:,end)), n_itrs1, ifftshift(img3==0), n_gen2, n_itrs2, n_cpys, [], n_conv1, cutoff1, cutoff2);
    case 9,
        rs = ghio2d( ifftshift(sqrt(img3)), pat > pat_cutoff, n_itrs, n_gen1, n_cpys,  ifftshift(img3==0), [] );      
    otherwise
        error('no reconstruction recipe specified');
end





% [rs, sup, M]...
%     = shrinkwrap( ifftshift( sqrt( img3 ) ),...
%     50, ifftshift( sqrt( img3 ) ) == 0, 40, 50, [], 3, 0.03, 0.06);
% rs = gshrinkwrap(  ifftshift( sqrt( img3 )     ), 50, ifftshift(img3==0), 20, 50, 4, [], 3, 0.03, 0.03);

% Sup = fftshift(ifft2(ifftshift( img3) , 'symmetric'));
% Sup = Sup > 0.01;
% rs = hio2d( ifftshift( sqrt(img3) ), Sup, 2000, ifftshift( sqrt(img3) )== 0);
% 






rs = beta1^-0.5 * rs;

figure(103),
imagesc( log10( beta1*abs( fftshift( fft2(rs(:,:,end) ) ) ).^2 ) );
axis image;
colormap;
colorbar;
hgsave([dir '\recon_log\' int2str(run) '\' int0str(tag,3 ) '\' datetime_string '_fft']);
caxis([0 5])

x = 0:dx:dx*(length(rs)-1);
x = 1e9 * x;

rs_end = rs(:,:,end)./(dx*1e9)^3;
rs_end( ~Sup(:,:,end) ) = 0;

figure(104),
imagesc( x, x, rs_end );
% imagesc(rs(:,:,end) );
axis image;
colormap(jet2);
colorbar;
caxis([0 max(rs_end(:))]);
hgsave([dir '\recon_log\' int2str(run) '\' int0str(tag,3 ) '\' datetime_string '_img']);

figure(105),
subplot(1, 2, 1);
imagesc( log10( beta1*abs( fftshift( fft2(rs(:,:,end) ) ) ).^2 ) );
axis image;
colormap;
colorbar;
title([int2str(run) '-' int2str(tag)]);
subplot(1, 2, 2);
imagesc( x, x, rs_end );
axis image;
colormap(jet2);
colorbar;
caxis([0 max(rs_end(:))]);
print('-dpng', [dir '\recon_log\' int2str(run) '\' int0str(tag,3 ) '\' datetime_string '.png']);
close(105);
    
 



