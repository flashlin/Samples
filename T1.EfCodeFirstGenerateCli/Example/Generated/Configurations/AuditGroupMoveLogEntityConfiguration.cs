using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AuditGroupMoveLogEntityConfiguration : IEntityTypeConfiguration<AuditGroupMoveLogEntity>
    {
        public void Configure(EntityTypeBuilder<AuditGroupMoveLogEntity> builder)
        {
            builder.ToTable("AuditGroupMoveLog");


            builder.Property(x => x.MoveDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.ChildAuditGroupId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.IomCustomerMappingCount)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.SboCustomerMappingCount)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.PreviousParentAuditGroupId)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.NewParentAuditGroupId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.CreatedBy)
                .HasColumnType("nvarchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
