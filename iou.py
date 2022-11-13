class Iouloss(nn.Module):
    def __init__(self):
        super(Iouloss, self).__init__()



    def trangle_area(self,a, b, c):
        return torch.abs((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
                (b[0] - c[0])) / 2.0

    def rbbox_to_corners(self,rbbox):
        # generate clockwise corners and rotate it clockwise
        # 顺时针方向返回角点位置
        cx, cy, x_d, y_d, angle = rbbox
        a_cos = torch.cos(angle)
        a_sin = torch.sin(angle)
        corners_x = [-x_d / 2, -x_d / 2, x_d / 2, x_d / 2]
        corners_y = [-y_d / 2, y_d / 2, y_d / 2, -y_d / 2]
        corners = [0] * 8
        for i in range(4):
            corners[2 *
                    i] = a_cos * corners_x[i] + \
                        a_sin * corners_y[i] + cx
            corners[2 * i +
                    1] = -a_sin * corners_x[i] + \
                        a_cos * corners_y[i] + cy
        return corners
    # 点在四边形(矩形)内?
    def point_in_quadrilateral(self,pt_x, pt_y, corners):
        a=(self.trangle_area([pt_x, pt_y], [corners[0],corners[1]] , [corners[2],corners[3]]) +
        self.trangle_area([pt_x, pt_y], [corners[4],corners[5]] , [corners[2],corners[3]]) +
        self.trangle_area([pt_x, pt_y], [corners[4],corners[5]] , [corners[6],corners[7]])+
        self.trangle_area([pt_x, pt_y], [corners[0],corners[1]] , [corners[6],corners[7]]))
        b=self.trangle_area([corners[4], corners[5]], [corners[0],corners[1]] , [corners[2],corners[3]])+self.trangle_area([corners[4], corners[5]], [corners[0],corners[1]] , [corners[6],corners[7]]) + 1e-4
        if a > b:
            return False
        else:
            return True 

    # 相交后转化为直线交点
    def line_segment_intersection(self,pts1, pts2, i, j):
        # pts1, pts2 为corners
        # i j 分别表示第几个交点，取其和其后一个点构成的线段
        # 返回为 tuple(bool, pts) bool=True pts为交点
        A, B, C, D, ret = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
        A[0] = pts1[2 * i]
        A[1] = pts1[2 * i + 1]

        B[0] = pts1[2 * ((i + 1) % 4)]
        B[1] = pts1[2 * ((i + 1) % 4) + 1]

        C[0] = pts2[2 * j]
        C[1] = pts2[2 * j + 1]

        D[0] = pts2[2 * ((j + 1) % 4)]
        D[1] = pts2[2 * ((j + 1) % 4) + 1]
        BA0 = B[0] - A[0]
        BA1 = B[1] - A[1]
        DA0 = D[0] - A[0]
        CA0 = C[0] - A[0]
        DA1 = D[1] - A[1]
        CA1 = C[1] - A[1]
        # 叉乘判断方向
        acd = DA1 * CA0 > CA1 * DA0
        bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
        if acd != bcd:
            abc = CA1 * BA0 > BA1 * CA0
            abd = DA1 * BA0 > BA1 * DA0
            # 判断方向
            if abc != abd:
                DC0 = D[0] - C[0]
                DC1 = D[1] - C[1]
                ABBA = A[0] * B[1] - B[0] * A[1]
                CDDC = C[0] * D[1] - D[0] * C[1]
                DH = BA1 * DC0 - BA0 * DC1
                Dx = ABBA * DC0 - BA0 * CDDC
                Dy = ABBA * DC1 - BA1 * CDDC
                ret[0] = Dx / DH
                ret[1] = Dy / DH
                return True, ret
        return False, ret

    # 顶点排序
    def sort_vertex_in_convex_polygon(self,int_pts, num_of_inter):
        def _cmp(pt, center):
            vx = pt[0] - center[0]
            vy = pt[1] - center[1]
            d = torch.sqrt(vx * vx + vy * vy)
            vx /= d
            vy /= d
            if vy < 0:
                vx = -2 - vx
            return vx

        if num_of_inter > 0:
            center = [0, 0]
            for i in range(num_of_inter):
                center[0] = center[0]+ int_pts[i][0]
                center[1] = center[1]+ int_pts[i][1]
            center[0] = center[0] /num_of_inter
            center[1] = center[1] /num_of_inter
            int_pts.sort(key=lambda x: _cmp(x, center))

    # 将多边形转化为多个三角形面积之和
    def area(self,int_pts, num_of_inter):
        

        area_val = torch.tensor([0]).to('cuda')
        for i in range(num_of_inter - 2):
            area_val = area_val+ abs(
                self.trangle_area(int_pts[0], int_pts[i + 1],
                            int_pts[i + 2]))
        return area_val
    
    
    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            a = pred.masked_select(mask).reshape(-1,10)
            b = target.masked_select(mask).reshape(-1,10)
            a = a[:,0:8]
            b = b[:,0:8]
            loss = torch.tensor([0]).to('cuda')
            for k in range(int(a.shape[0])):
                pre = a[k,:]
                tar = b[k,:]
                lambda0 = 3
                pts, num_pts = [], 0
                for i in range(4):
                    point = [pre[2 * i], pre[2 * i + 1]]
                    if self.point_in_quadrilateral(point[0], point[1],
                                            tar):
                        num_pts += 1
                        pts.append(point)
                for i in range(4):
                    point = [tar[2 * i], tar[2 * i + 1]]
                    if self.point_in_quadrilateral(point[0], point[1],
                                            pre):
                        num_pts += 1
                        pts.append(point)
                for i in range(4):
                    for j in range(4):
                        ret, point = self.line_segment_intersection(pre, tar, i, j)
                        if ret:
                            num_pts += 1
                            pts.append(point)
                self.sort_vertex_in_convex_polygon(pts, num_pts)
                intersection = self.area(pts, num_pts)
                pre_area = self.trangle_area([pre[0],pre[1]], [pre[2],pre[3]],[pre[4],pre[5]]) + self.trangle_area([pre[0],pre[1]], [pre[6],pre[7]],[pre[4],pre[5]])
                tar_area = self.trangle_area([tar[0],tar[1]], [tar[2],tar[3]],[tar[4],tar[5]]) + self.trangle_area([tar[0],tar[1]], [tar[6],tar[7]],[tar[4],tar[5]])
                iou = intersection/(pre_area + tar_area - intersection)
                loss = loss + (1 - torch.pow(iou,lambda0))
            return loss/int(a.shape[0])
        else:
            return 0.
